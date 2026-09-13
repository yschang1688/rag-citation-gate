"""GPU 對照組：在 Colab T4（免費額度）上用 vLLM 跑與 M1 完全相同的量測。

> **2026-09-13 已實跑驗證**（Colab 免費 T4，結果在 `bench/vllm_results.json`）。
> 腳本本體不用改，但環境要先修一個東西：**Colab 預裝的 `torchaudio` 與 vLLM
> 帶進來的 torch 是不同 CUDA 版本編譯的**，`_check_cuda_version()` 會在 import
> 階段拋錯，`vllm serve` 連 server 都起不來。安裝完先 `pip uninstall -y torchaudio`。
>
> 這支腳本有一個已知弱點沒有修（留著當教材）：等待迴圈只檢查端點通不通、
> **沒有檢查子行程是否還活著**，所以上面那個失敗是靜默的——server 早就死了，
> 迴圈還會安靜等滿 20 分鐘。要用在別的地方時，請加上 `p.poll() is not None` 的提早跳出。

## 怎麼用

1. 開 https://colab.research.google.com → 新增筆記本
2. 執行階段 → 變更執行階段類型 → T4 GPU
3. 把本檔全文貼進一個 cell，執行
4. 跑完會在左側檔案欄產生 `vllm_results.json`，下載後給我

## 為什麼值得跑這一趟

M1 那組量到的是「沒有 continuous batching 的樣子」：並發 4 時總吞吐比並發 1 還低、
TTFT 惡化 51 倍。vLLM 的 PagedAttention + continuous batching 正是為此而生。
跑完這組，你就有**同一套量測方法、兩種架構**的對照數據，面試講「為什麼需要 vLLM」
時說的是自己的數字，不是轉述。

## 已知的不對等（報告時必須標明，不可略過）

- **量化不同**：Ollama 的 qwen3.5:4b 是 Q4_K_M；本腳本預設跑 FP16。
  權重位元組數差約 4 倍，decode 速度不可直接相比。
- **硬體不同**：M1 是統一記憶體（頻寬約 68 GB/s），T4 是獨立 GPU（約 320 GB/s）。
- **可以直接比的是「形狀」不是「絕對值」**：TTFT 隨上下文的增長曲線、
  並發對總吞吐的影響方向。這兩件事才是這次對照要回答的。
"""

# ============================================================
# 以下整段貼進 Colab cell
# ============================================================
SETUP = r'''
!nvidia-smi
!pip install -q vllm
'''

BENCH = r'''
import json, statistics, subprocess, threading, time, urllib.request, os, signal

# ⚠ 若這個 repo 名在 2026-09 已變更，改這裡。挑與 M1 端相近大小的 Qwen 家族模型。
MODEL = "Qwen/Qwen2.5-3B-Instruct"
PORT  = 8000
BASE  = f"http://127.0.0.1:{PORT}"

# 與 serve_bench.py 逐字相同的語料與問句，確保長度軸可比
FILLER = ("銀行法第十二條所稱擔保授信，謂對於銀行之授信，提供左列之一為擔保者："
          "不動產或動產抵押權、動產或權利質權、借款人營業交易所發生之應收票據、"
          "各級政府公庫主管機關、銀行或經政府核准設立之信用保證機構之保證。")
QUESTION = "\n\n根據以上內容，用一句話說明什麼是擔保授信。"

def make_prompt(n):
    if n <= 0: return QUESTION.strip()
    return FILLER * max(1, n // len(FILLER)) + QUESTION

# ---- 啟動 vLLM 的 OpenAI 相容 server（背景），用 HTTP 量測以對齊 Ollama 那側 ----
# T4 是 Turing，不支援 bfloat16，必須指定 float16。
proc = subprocess.Popen(
    ["vllm", "serve", MODEL, "--port", str(PORT), "--dtype", "float16",
     "--max-model-len", "8192", "--gpu-memory-utilization", "0.90"],
    stdout=open("vllm.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid)

print("等待 server 就緒（首次要下載權重，可能數分鐘）…")
for i in range(600):
    try:
        urllib.request.urlopen(f"{BASE}/v1/models", timeout=2); print("✓ 就緒"); break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit("server 未就緒，看 vllm.log")

def post_stream(prompt, num_predict):
    """回傳 (ttft_s, total_s, out_chunks)。TTFT 以第一個內容 chunk 的牆鐘時間為準——
    與 M1 端同一個定義。"""
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "max_tokens": num_predict,
        "temperature": 0, "stream": True,
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/completions", data=body,
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter(); ttft = None; n = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"): continue
            payload = line[5:].strip()
            if payload == "[DONE]": break
            try: txt = json.loads(payload)["choices"][0].get("text", "")
            except Exception: continue
            if txt:
                if ttft is None: ttft = time.perf_counter() - t0
                n += 1
    return (ttft or time.perf_counter() - t0), time.perf_counter() - t0, n

out = {"meta": {"host": "Colab T4 16GB", "engine": "vLLM (PagedAttention + continuous batching)",
                "model": MODEL, "dtype": "float16",
                "note": "與 M1/Ollama 端同一套量測方法；量化與硬體不同，比較的是形狀不是絕對值。",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
       "latency": [], "concurrency": []}

print("\n=== 延遲 / 吞吐 vs 上下文長度 ===")
for target in [0, 400, 1600, 4000]:
    p = make_prompt(target)
    post_stream(p, 8)                                   # warm-up
    s = [post_stream(p, 64) for _ in range(3)]
    ttfts = [x[0] for x in s]
    tps = [(x[2] - 1) / (x[1] - x[0]) if x[1] > x[0] and x[2] > 1 else 0 for x in s]
    row = {"model": MODEL, "target_prompt_tokens": target,
           "ttft_s_median": round(statistics.median(ttfts), 3),
           "decode_tok_s_median": round(statistics.median(tps), 2)}
    out["latency"].append(row)
    print(f"  target={target:>5} | TTFT {row['ttft_s_median']:>6.3f}s | decode {row['decode_tok_s_median']:>6.2f} tok/s")

print("\n=== 並發（重點：看總吞吐會不會隨並發上升）===")
prompt = make_prompt(400)
post_stream(prompt, 8)
for conc in [1, 2, 4, 8]:
    res, lock = [], threading.Lock()
    def worker():
        r = post_stream(prompt, 48)
        with lock: res.append(r)
    ths = [threading.Thread(target=worker) for _ in range(conc)]
    t0 = time.perf_counter()
    [t.start() for t in ths]; [t.join() for t in ths]
    wall = time.perf_counter() - t0
    row = {"model": MODEL, "concurrency": conc, "wall_s": round(wall, 2),
           "aggregate_tok_s": round(sum(r[2] for r in res) / wall, 2),
           "ttft_s_median": round(statistics.median([r[0] for r in res]), 3),
           "ttft_s_max": round(max(r[0] for r in res), 3)}
    out["concurrency"].append(row)
    print(f"  conc={conc:>2} | 總吞吐 {row['aggregate_tok_s']:>7.2f} tok/s | "
          f"TTFT 中位 {row['ttft_s_median']:>6.3f}s 最差 {row['ttft_s_max']:>6.3f}s")

with open("vllm_results.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\n✓ 已寫入 vllm_results.json —— 下載這個檔案")

os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
'''

# ============================================================
# 並發飽和點掃描：第一輪只測到 8，吞吐還在爬，沒看到轉折。
#
# 「吞吐隨並發線性上升」在 1–8 成立，但那不是結論而是「還沒到頭」——
# 任何排隊系統都有飽和點，只是位置沒量到。這一支把並發推到 32，
# 要回答的是：**吞吐在哪裡不再上升，以及那一刻延遲付出多少代價。**
#
# 與 BENCH 的差異：
#   - 只掃並發（不重跑上下文長度那軸，那條已經有答案）
#   - 每格跑兩輪取較好值，因為高並發的單輪變異比低並發大
#   - 多記 per-request 延遲與 p95，飽和之後「平均還行但尾巴很慘」是常態
#   - 等待迴圈**同時檢查子行程死活**，修掉 BENCH 那個靜默失敗的弱點
# ============================================================
CONC_SWEEP = r'''
import json, statistics, subprocess, threading, time, urllib.request, os, signal

MODEL = "Qwen/Qwen2.5-3B-Instruct"
PORT  = 8000
BASE  = f"http://127.0.0.1:{PORT}"
LEVELS = [1, 2, 4, 8, 16, 32]
ROUNDS = 2

FILLER = ("銀行法第十二條所稱擔保授信，謂對於銀行之授信，提供左列之一為擔保者："
          "不動產或動產抵押權、動產或權利質權、借款人營業交易所發生之應收票據、"
          "各級政府公庫主管機關、銀行或經政府核准設立之信用保證機構之保證。")
QUESTION = "\n\n根據以上內容，用一句話說明什麼是擔保授信。"
PROMPT = FILLER * max(1, 400 // len(FILLER)) + QUESTION

def server_up():
    try:
        urllib.request.urlopen(f"{BASE}/v1/models", timeout=2)
        return True
    except Exception:
        return False

proc = None
if server_up():
    print("✓ 沿用既有 server")
else:
    proc = subprocess.Popen(
        ["vllm", "serve", MODEL, "--port", str(PORT), "--dtype", "float16",
         "--max-model-len", "8192", "--gpu-memory-utilization", "0.90"],
        stdout=open("vllm.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    print("啟動 server…")
    for i in range(600):
        # 同時檢查「行程還活著嗎」——只檢查端點的話，行程早就死了還會安靜等滿 20 分鐘
        if proc.poll() is not None:
            print(open("vllm.log").read()[-2000:])
            raise SystemExit(f"server 在啟動時就掛了（exit {proc.returncode}），日誌如上")
        if server_up():
            print("✓ 就緒"); break
        time.sleep(2)
    else:
        raise SystemExit("server 逾時未就緒，看 vllm.log")

def post_stream(prompt, num_predict):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": num_predict,
                       "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(f"{BASE}/v1/completions", data=body,
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter(); ttft = None; n = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"): continue
            payload = line[5:].strip()
            if payload == "[DONE]": break
            try: txt = json.loads(payload)["choices"][0].get("text", "")
            except Exception: continue
            if txt:
                if ttft is None: ttft = time.perf_counter() - t0
                n += 1
    return (ttft or time.perf_counter() - t0), time.perf_counter() - t0, n

def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]

def run(conc):
    res, lock = [], threading.Lock()
    def worker():
        r = post_stream(PROMPT, 48)
        with lock: res.append(r)
    ths = [threading.Thread(target=worker) for _ in range(conc)]
    t0 = time.perf_counter()
    [t.start() for t in ths]; [t.join() for t in ths]
    wall = time.perf_counter() - t0
    ttfts = [r[0] for r in res]
    return {"concurrency": conc, "wall_s": round(wall, 2),
            "aggregate_tok_s": round(sum(r[2] for r in res) / wall, 2),
            "ttft_s_median": round(statistics.median(ttfts), 3),
            "ttft_s_p95": round(pct(ttfts, 0.95), 3),
            "ttft_s_max": round(max(ttfts), 3),
            "e2e_s_median": round(statistics.median([r[1] for r in res]), 3),
            "completed": len(res)}

post_stream(PROMPT, 8)  # warm-up
out = {"meta": {"host": "Colab T4 16GB", "engine": "vLLM", "model": MODEL,
                "dtype": "float16", "levels": LEVELS, "rounds": ROUNDS,
                "prompt_target_tokens": 400, "max_tokens": 48,
                "note": "找吞吐飽和點。每格兩輪取吞吐較高者，高並發單輪變異較大。",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
       "sweep": []}

print("\n=== 並發飽和點掃描 ===")
prev = None
for c in LEVELS:
    best = max((run(c) for _ in range(ROUNDS)), key=lambda r: r["aggregate_tok_s"])
    gain = None if prev is None else round(best["aggregate_tok_s"] / prev, 2)
    best["gain_vs_prev"] = gain
    prev = best["aggregate_tok_s"]
    out["sweep"].append(best)
    g = "—" if gain is None else f"x{gain}"
    print(f"  conc={c:>3} | 吞吐 {best['aggregate_tok_s']:>8.2f} tok/s ({g:>6}) | "
          f"TTFT 中位 {best['ttft_s_median']:>6.3f} p95 {best['ttft_s_p95']:>6.3f} "
          f"最差 {best['ttft_s_max']:>6.3f} | 端到端中位 {best['e2e_s_median']:>6.3f}")

# 飽和點的判準寫在程式裡，不靠事後目測：相對前一格的增幅首次掉到 1.2 倍以下
# （並發翻倍、吞吐卻不到 1.2 倍，代表已經不是靠批次化在擴充）
sat = next((r["concurrency"] for r in out["sweep"]
            if r["gain_vs_prev"] is not None and r["gain_vs_prev"] < 1.2), None)
out["meta"]["saturation_at"] = sat
out["meta"]["saturation_rule"] = "併發翻倍時吞吐增幅首次 < 1.2x"
print(f"\n飽和點（增幅首次 < 1.2x）：{sat if sat else '未在 32 以內出現'}")

with open("vllm_conc_sweep.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("✓ 已寫入 vllm_conc_sweep.json")

if proc is not None:
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
'''

# ============================================================
# 歸因探針：那個飽和點是伺服器的，還是我自己的客戶端的？
#
# CONC_SWEEP 在並發 192 量到吞吐增幅掉到 1.09，看起來是飽和。
# 但壓力是用 192 條 Python 執行緒打出去的，每條都在解析 SSE 事件流，
# **而 GIL 會把那些解析序列化**。如果客戶端才是瓶頸，量到的「飽和」
# 就是我自己的量測工具的極限，跟 vLLM 無關——這種結論會很體面地錯下去。
#
# 判準（在持續高並發負載下同時採樣兩邊）：
#   GPU 使用率貼近 100%  → 伺服器真的吃滿了，飽和點成立
#   GPU 使用率明顯偏低   → 瓶頸在客戶端，那個數字不能當 vLLM 的上限報
#
# 這支與 completion-gate 的「量測管線自己要被驗證」是同一件事：
# 先證明工具沒有先壞掉，才有資格解讀它吐出來的數字。
# ============================================================
GPU_PROBE = r'''
import json, os, signal, statistics, subprocess, threading, time, urllib.request

MODEL = "Qwen/Qwen2.5-3B-Instruct"
PORT  = 8000
BASE  = f"http://127.0.0.1:{PORT}"
CONC  = 192
HOLD_S = 25

FILLER = ("銀行法第十二條所稱擔保授信，謂對於銀行之授信，提供左列之一為擔保者："
          "不動產或動產抵押權、動產或權利質權、借款人營業交易所發生之應收票據、"
          "各級政府公庫主管機關、銀行或經政府核准設立之信用保證機構之保證。")
PROMPT = FILLER * max(1, 400 // len(FILLER)) + "\n\n根據以上內容，用一句話說明什麼是擔保授信。"

def server_up():
    try:
        urllib.request.urlopen(f"{BASE}/v1/models", timeout=2); return True
    except Exception:
        return False

proc = None
if server_up():
    print("✓ 沿用既有 server")
else:
    proc = subprocess.Popen(
        ["vllm", "serve", MODEL, "--port", str(PORT), "--dtype", "float16",
         "--max-model-len", "8192", "--gpu-memory-utilization", "0.90"],
        stdout=open("vllm.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    print("啟動 server…")
    for i in range(600):
        if proc.poll() is not None:
            print(open("vllm.log").read()[-2000:])
            raise SystemExit(f"server 啟動即掛（exit {proc.returncode}）")
        if server_up():
            print("✓ 就緒"); break
        time.sleep(2)
    else:
        raise SystemExit("server 逾時未就緒")

def one_request():
    body = json.dumps({"model": MODEL, "prompt": PROMPT, "max_tokens": 48,
                       "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(f"{BASE}/v1/completions", data=body,
                                 headers={"content-type": "application/json"})
    n = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                n += 1
    return n

stop = threading.Event()
counts = []
clock = threading.Lock()

def loader():
    local = 0
    while not stop.is_set():
        try: local += one_request()
        except Exception: pass
    with clock: counts.append(local)

def cpu_ticks():
    # 本行程與所有子執行緒累計的 CPU 時間（utime+stime），單位是 clock tick
    with open("/proc/self/stat") as f:
        p = f.read().split()
    return int(p[13]) + int(p[14])

def gpu_util():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5).stdout.strip()
        a, b = o.split(",")
        return int(a), int(b)
    except Exception:
        return None, None

one_request()  # warm-up
ths = [threading.Thread(target=loader, daemon=True) for _ in range(CONC)]
print(f"\n持續打 {CONC} 並發 {HOLD_S} 秒，同時採樣 GPU 與客戶端 CPU…")
t0 = time.perf_counter(); c0 = cpu_ticks()
[t.start() for t in ths]
utils, mems = [], []
time.sleep(3)  # 先讓佇列填滿再採樣，避免把爬升期算進去
while time.perf_counter() - t0 < HOLD_S:
    u, mb = gpu_util()
    if u is not None: utils.append(u); mems.append(mb)
    time.sleep(0.5)
stop.set()
[t.join(timeout=60) for t in ths]
wall = time.perf_counter() - t0
cpu_s = (cpu_ticks() - c0) / os.sysconf("SC_CLK_TCK")

ncpu = os.cpu_count()
res = {"concurrency": CONC, "hold_s": round(wall, 1),
       "gpu_util_median": statistics.median(utils) if utils else None,
       "gpu_util_min": min(utils) if utils else None,
       "gpu_util_max": max(utils) if utils else None,
       "gpu_mem_used_mb_max": max(mems) if mems else None,
       "samples": len(utils),
       "client_cpu_core_equivalent": round(cpu_s / wall, 2),
       "client_cpu_cores_available": ncpu,
       "client_cpu_saturation_pct": round(100 * (cpu_s / wall) / ncpu, 1),
       "aggregate_tok_s": round(sum(counts) / wall, 2)}

print(json.dumps(res, ensure_ascii=False, indent=1))
gm = res["gpu_util_median"]
cc = res["client_cpu_core_equivalent"]
print()
if gm is not None and gm >= 90:
    print(f"→ GPU 中位使用率 {gm}%：**伺服器吃滿了**，飽和點是 vLLM 這側的，結論成立。")
elif gm is not None and cc >= 0.9 * ncpu:
    print(f"→ GPU 只有 {gm}% 而客戶端 CPU 已佔滿 {cc}/{ncpu} 核："
          f"**瓶頸在量測客戶端，不是 vLLM**。那個飽和點不能當伺服器上限報。")
else:
    print(f"→ GPU {gm}%、客戶端 {cc}/{ncpu} 核：兩邊都沒吃滿，瓶頸可能在單請求的往返延遲，"
          f"需要再查（例如連線數上限或 HTTP keep-alive）。")

with open("vllm_gpu_probe.json", "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("✓ 已寫入 vllm_gpu_probe.json")

if proc is not None:
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
'''

# ============================================================
# 量化對比 ＋ Temperature=0 非確定性：一次 exec 全自動跑完兩個實驗。
#
# ① 量化（FP16 vs AWQ 4-bit，同一張 T4、同一套量測）
#    有一個事前寫下的可證偽預測：decode 是 memory-bound（M1 那組的發現二），
#    所以 4-bit 的 decode 加速「理論上限」是位元組比 4 倍，實務預期 2–3 倍
#    （反量化開銷＋activation 仍是 FP16 要吃頻寬）。
#    - 顯著低於 2 倍 → 反量化開銷吃掉頻寬紅利，這本身就是發現
#    - 接近甚至超過 4 倍 → 量測有問題，先懷疑自己
#    順帶記兩個旁證：兩模型就緒後的顯存佔用（權重腳印）、以及 5 題
#    temperature=0 的回答是否逐字相同（**煙霧級**品質檢查——只能說
#    「輸出有沒有變」，不能說「精度掉多少」，那需要評估集，仍列未測）。
#
# ② Temperature=0 非確定性（演練包第二關目前是純理論句，補一手數據）
#    同一 prompt、temperature=0、各 12 次：
#    A 條件「單獨送」（無其他負載）vs B 條件「混在雜訊並發裡送」
#    （6 條背景執行緒打長短不一的隨機 prompt，讓目標請求每次和
#    不同鄰居被 batch 在一起）。比對輸出雜湊的相異數。
#    預測：A 應該 1 種；B 若 >1 種，「batch 組成影響輸出」就從轉述變實測。
#    B 若也是 1 種，結論是「此規模觀察不到」——邊界也是收穫。
# ============================================================
QUANT_DET = r'''
import hashlib, json, os, random, signal, statistics, subprocess, threading, time, urllib.request

PORT = 8000
BASE = f"http://127.0.0.1:{PORT}"
FP16 = "Qwen/Qwen2.5-3B-Instruct"
AWQ  = "Qwen/Qwen2.5-3B-Instruct-AWQ"

FILLER = ("銀行法第十二條所稱擔保授信，謂對於銀行之授信，提供左列之一為擔保者："
          "不動產或動產抵押權、動產或權利質權、借款人營業交易所發生之應收票據、"
          "各級政府公庫主管機關、銀行或經政府核准設立之信用保證機構之保證。")
QUESTION = "\n\n根據以上內容，用一句話說明什麼是擔保授信。"

def make_prompt(n):
    if n <= 0: return QUESTION.strip()
    return FILLER * max(1, n // len(FILLER)) + QUESTION

def gpu_mem_mb():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5).stdout.strip()
        return int(o)
    except Exception:
        return None

def start(model):
    proc = subprocess.Popen(
        ["vllm", "serve", model, "--port", str(PORT), "--dtype", "float16",
         "--max-model-len", "8192", "--gpu-memory-utilization", "0.90"],
        stdout=open("vllm.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    print(f"啟動 {model} …")
    for i in range(900):
        if proc.poll() is not None:
            print(open("vllm.log").read()[-2500:])
            raise SystemExit(f"{model} 啟動即掛（exit {proc.returncode}）")
        try:
            urllib.request.urlopen(f"{BASE}/v1/models", timeout=2)
            print("✓ 就緒"); return proc
        except Exception:
            time.sleep(2)
    raise SystemExit("逾時未就緒")

def stop(proc):
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    for _ in range(90):
        if proc.poll() is not None: break
        time.sleep(1)
    time.sleep(8)   # 等顯存真的還回來再起下一個

def post_stream(prompt, num_predict):
    body = json.dumps({"model": CUR, "prompt": prompt, "max_tokens": num_predict,
                       "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(f"{BASE}/v1/completions", data=body,
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter(); ttft = None; n = 0; parts = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"): continue
            payload = line[5:].strip()
            if payload == "[DONE]": break
            try: txt = json.loads(payload)["choices"][0].get("text", "")
            except Exception: continue
            if txt:
                if ttft is None: ttft = time.perf_counter() - t0
                n += 1; parts.append(txt)
    return (ttft or time.perf_counter() - t0), time.perf_counter() - t0, n, "".join(parts)

def speed():
    rows = []
    for target in [0, 400, 1600, 4000]:
        p = make_prompt(target)
        post_stream(p, 8)                                    # warm-up
        s = [post_stream(p, 64) for _ in range(3)]
        tps = [(x[2] - 1) / (x[1] - x[0]) if x[1] > x[0] and x[2] > 1 else 0 for x in s]
        rows.append({"target_prompt_tokens": target,
                     "ttft_s_median": round(statistics.median(x[0] for x in s), 3),
                     "decode_tok_s_median": round(statistics.median(tps), 2)})
        print(f"  target={target:>5} | TTFT {rows[-1]['ttft_s_median']:>6.3f}s "
              f"| decode {rows[-1]['decode_tok_s_median']:>6.2f} tok/s")
    return rows

QUALITY_QS = ["根據以上內容，用一句話說明什麼是擔保授信。",
              "根據以上內容，列出可作為擔保的項目。",
              "根據以上內容，誰可以擔任保證機構？",
              "根據以上內容，應收票據要符合什麼條件才能當擔保？",
              "根據以上內容，動產可以用哪些方式設定擔保？"]

def answers():
    out = {}
    for q in QUALITY_QS:
        out[q] = post_stream(FILLER + "\n\n" + q, 96)[3]
    return out

def determinism(n_rep=12, noise_threads=6):
    prompt = make_prompt(400)
    print("  A 條件：單獨送（無其他負載）…")
    iso = [post_stream(prompt, 64)[3] for _ in range(n_rep)]
    print("  B 條件：混在雜訊並發裡送…")
    stop_ev = threading.Event()
    def noise(seed):
        rnd = random.Random(seed)
        while not stop_ev.is_set():
            n = rnd.choice([50, 300, 900, 2000])
            try: post_stream(make_prompt(n) + f"（附註 {rnd.randint(0, 9999)}）", 32)
            except Exception: pass
    ths = [threading.Thread(target=noise, args=(i,), daemon=True) for i in range(noise_threads)]
    [t.start() for t in ths]
    time.sleep(3)
    mix = [post_stream(prompt, 64)[3] for _ in range(n_rep)]
    stop_ev.set(); [t.join(timeout=60) for t in ths]
    h = lambda x: hashlib.sha1(x.encode()).hexdigest()[:10]
    hi, hm = sorted({h(x) for x in iso}), sorted({h(x) for x in mix})
    div = next((x for x in mix if h(x) != h(iso[0])), None)
    return {"n_rep": n_rep, "noise_threads": noise_threads,
            "isolated_distinct": len(hi), "mixed_distinct": len(hm),
            "isolated_hashes": hi, "mixed_hashes": hm,
            "baseline_output": iso[0],
            "divergent_output_example": div}

res = {"meta": {"host": "Colab T4 16GB", "engine": "vLLM", "dtype": "float16",
                "fp16_model": FP16, "awq_model": AWQ,
                "prediction_written_before_run": "decode 加速理論上限 4x（位元組比），實務預期 2–3x；顯著低於 2x = 反量化開銷吃掉頻寬紅利",
                "quality_note": "5 題回答比對是煙霧級檢查，只能說輸出有無改變，不是精度評估——精度對比仍列未測",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}}

CUR = FP16
p = start(FP16)
res["fp16_gpu_mem_mb"] = gpu_mem_mb()
print(f"\n=== FP16 速度（顯存 {res['fp16_gpu_mem_mb']} MB）===")
res["fp16_speed"] = speed()
print("\n=== Temperature=0 非確定性（FP16 server 上直接做）===")
res["determinism"] = determinism()
d = res["determinism"]
print(f"  單獨送：{d['isolated_distinct']} 種輸出；混雜送：{d['mixed_distinct']} 種輸出")
res["fp16_answers"] = answers()
stop(p)

CUR = AWQ
p = start(AWQ)
res["awq_gpu_mem_mb"] = gpu_mem_mb()
print(f"\n=== AWQ 速度（顯存 {res['awq_gpu_mem_mb']} MB）===")
res["awq_speed"] = speed()
res["awq_answers"] = answers()
stop(p)

print("\n=== 量化對比 ===")
cmp_rows = []
for a, b in zip(res["fp16_speed"], res["awq_speed"]):
    r = {"target_prompt_tokens": a["target_prompt_tokens"],
         "decode_speedup": round(b["decode_tok_s_median"] / a["decode_tok_s_median"], 2)
                           if a["decode_tok_s_median"] else None,
         "ttft_ratio": round(b["ttft_s_median"] / a["ttft_s_median"], 2)
                       if a["ttft_s_median"] else None}
    cmp_rows.append(r)
    print(f"  target={r['target_prompt_tokens']:>5} | decode 加速 x{r['decode_speedup']} | TTFT 比 x{r['ttft_ratio']}")
res["comparison"] = cmp_rows
same = sum(res["fp16_answers"][q] == res["awq_answers"][q] for q in QUALITY_QS)
res["answers_identical"] = f"{same}/{len(QUALITY_QS)}"
print(f"  5 題 temperature=0 回答逐字相同：{res['answers_identical']}")
print(f"  顯存腳印：FP16 {res['fp16_gpu_mem_mb']} MB vs AWQ {res['awq_gpu_mem_mb']} MB")

with open("vllm_quant_det.json", "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("\n✓ 已寫入 vllm_quant_det.json")
'''

# ============================================================
# KV cache 顯存量測：把「上下文的代價在 KV cache、不在權重」量出來。
#
# 上一輪（QUANT_DET）留下一個只推論、未證實的說法：FP16 與 AWQ 的
# nvidia-smi 腳印幾乎相同，我推論是「vLLM 吃滿預算，權重省的空間變成
# KV cache」。這一支從 vLLM 自己回報的 KV 池大小正面驗證——推論不能
# 一直當結論用。
#
# 三個事前寫下的預測（Qwen2.5-3B：36 層 × 2 KV heads × 128 dim × K+V
# × FP16 = 每 token 36,864 bytes ≈ 36 KB，由 config.json 算出）：
#   P1 同模型，gpu-memory-utilization 0.5→0.7→0.9：KV 池 token 數應
#      隨 util 線性增加，斜率 ≈ T4 總顯存 × 0.2 ÷ 36KB（每格約 8 萬 token）
#   P2 同 util=0.9，AWQ 的 KV 池應比 FP16 大 ≈ 權重差（約 3.7GB）÷ 36KB
#      ≈ 10 萬 token——這一格直接證實上輪的推論
#   P3 動態佔用：decode 進行中 /metrics 的 gpu_cache_usage_perc × 池大小
#      應 ≈ 該請求的 prompt+已生成 token 數（block 粒度 16 造成的誤差內）
# ============================================================
KV_PROBE = r'''
import json, os, re, signal, subprocess, threading, time, urllib.request

PORT = 8000
BASE = f"http://127.0.0.1:{PORT}"
FP16 = "Qwen/Qwen2.5-3B-Instruct"
AWQ  = "Qwen/Qwen2.5-3B-Instruct-AWQ"
KV_BYTES_PER_TOKEN = 36864   # 由 config.json 算出：2(K+V)×2 heads×128 dim×2 bytes×36 layers

FILLER = ("銀行法第十二條所稱擔保授信，謂對於銀行之授信，提供左列之一為擔保者："
          "不動產或動產抵押權、動產或權利質權、借款人營業交易所發生之應收票據、"
          "各級政府公庫主管機關、銀行或經政府核准設立之信用保證機構之保證。")

def make_prompt(n):
    return FILLER * max(1, n // len(FILLER)) + "\n\n根據以上內容，用一句話說明什麼是擔保授信。"

def start(model, util, tag):
    log = f"vllm_{tag}.log"
    proc = subprocess.Popen(
        ["vllm", "serve", model, "--port", str(PORT), "--dtype", "float16",
         "--max-model-len", "8192", "--gpu-memory-utilization", str(util)],
        stdout=open(log, "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    print(f"啟動 {model} util={util} …")
    for i in range(900):
        if proc.poll() is not None:
            print(open(log).read()[-2000:])
            raise SystemExit(f"{tag} 啟動即掛（exit {proc.returncode}）")
        try:
            urllib.request.urlopen(f"{BASE}/v1/models", timeout=2)
            print("✓ 就緒"); return proc, log
        except Exception:
            time.sleep(2)
    raise SystemExit("逾時未就緒")

def stop(proc):
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    for _ in range(90):
        if proc.poll() is not None: break
        time.sleep(1)
    time.sleep(8)

def kv_from_log(log):
    """vLLM 啟動 log 會自己報 KV 池大小與最大並發，抓那兩行。"""
    t = open(log).read()
    m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", t)
    c = re.search(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x", t)
    return {"kv_pool_tokens": int(m.group(1).replace(",", "")) if m else None,
            "max_concurrency_at_max_len": float(c.group(2)) if c else None}

def cache_usage():
    try:
        t = urllib.request.urlopen(f"{BASE}/metrics", timeout=5).read().decode()
        m = [float(x) for x in re.findall(r'vllm:gpu_cache_usage_perc\S*\s+([\d.eE+-]+)', t)]
        return max(m) if m else None
    except Exception:
        return None

def post(prompt, num_predict):
    body = json.dumps({"model": CUR, "prompt": prompt, "max_tokens": num_predict,
                       "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(f"{BASE}/v1/completions", data=body,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        for _ in r: pass

res = {"meta": {"host": "Colab T4 16GB", "engine": "vLLM", "dtype": "float16",
                "kv_bytes_per_token_theory": KV_BYTES_PER_TOKEN,
                "predictions_written_before_run": [
                 "P1: KV 池 token 數隨 util 線性增加，每 +0.2 util ≈ +8 萬 token",
                 "P2: 同 util=0.9，AWQ 的 KV 池比 FP16 大 ≈ 權重差 3.7GB ÷ 36KB ≈ 10 萬 token",
                 "P3: decode 中 usage × 池大小 ≈ 該請求的 token 數（block 粒度誤差內）"],
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
       "static": [], "dynamic": []}

CUR = FP16
for util in (0.5, 0.7, 0.9):
    p, log = start(FP16, util, f"fp16_u{util}")
    row = {"model": "FP16", "util": util, **kv_from_log(log)}
    row["kv_pool_gb_implied"] = (round(row["kv_pool_tokens"] * KV_BYTES_PER_TOKEN / 2**30, 2)
                                 if row["kv_pool_tokens"] else None)
    res["static"].append(row)
    print(f"  FP16 util={util} | KV 池 {row['kv_pool_tokens']} tokens "
          f"≈ {row['kv_pool_gb_implied']} GB | 8K 滿載最大並發 {row['max_concurrency_at_max_len']}x")
    if util == 0.9:
        # P3 動態量測：decode 進行中讀 /metrics，比對 usage×池 與請求 token 數
        print("  動態佔用（decode 進行中採樣 /metrics）…")
        for target in (400, 1600, 4000):
            peak = {"v": None}
            def watch():
                t0 = time.time()
                while time.time() - t0 < 30:
                    u = cache_usage()
                    if u is not None:
                        peak["v"] = max(peak["v"] or 0, u)
                    time.sleep(0.25)
            w = threading.Thread(target=watch, daemon=True); w.start()
            post(make_prompt(target), 64)
            time.sleep(0.5)
            used = (round(peak["v"] * row["kv_pool_tokens"]) if peak["v"] else None)
            d = {"target_prompt_tokens": target, "peak_usage_perc": peak["v"],
                 "implied_tokens_in_cache": used}
            res["dynamic"].append(d)
            print(f"    target={target:>5} | 峰值 usage {peak['v']} | 折算 {used} tokens")
    stop(p)

CUR = AWQ
p, log = start(AWQ, 0.9, "awq_u0.9")
row = {"model": "AWQ", "util": 0.9, **kv_from_log(log)}
row["kv_pool_gb_implied"] = (round(row["kv_pool_tokens"] * KV_BYTES_PER_TOKEN / 2**30, 2)
                             if row["kv_pool_tokens"] else None)
res["static"].append(row)
print(f"  AWQ  util=0.9 | KV 池 {row['kv_pool_tokens']} tokens "
      f"≈ {row['kv_pool_gb_implied']} GB | 8K 滿載最大並發 {row['max_concurrency_at_max_len']}x")
stop(p)

fp09 = next(r for r in res["static"] if r["model"] == "FP16" and r["util"] == 0.9)
res["p2_awq_minus_fp16_tokens"] = (row["kv_pool_tokens"] - fp09["kv_pool_tokens"]
                                   if row["kv_pool_tokens"] and fp09["kv_pool_tokens"] else None)
print(f"\nP2 驗證：AWQ 池 − FP16 池 = {res['p2_awq_minus_fp16_tokens']} tokens "
      f"≈ {round(res['p2_awq_minus_fp16_tokens']*KV_BYTES_PER_TOKEN/2**30,2) if res['p2_awq_minus_fp16_tokens'] else '?'} GB（預測 ≈ 權重差 3.7GB）")

with open("vllm_kv_probe.json", "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("✓ 已寫入 vllm_kv_probe.json")
'''

# ============================================================
# KV_PROBE 的救援解析：第一輪的 regex 沒對上這版 vLLM 的 log 字樣，
# 四格 static 全拿 None。但四個 log 檔都留在磁碟上——資料沒丟，
# 只是解析錯了，所以不用重跑 server，在 kernel 端重新解析即可。
# 這支同時把 P3（動態量測）補掉：重啟一台 FP16@0.9，先從 /metrics
# 全文自動探測含 cache 與 usage 的 metric 名，再做採樣——不再猜字樣。
# 印出的每一行都先清成安全字元集，避免瀏覽器端的內容過濾把結果吃掉。
# ============================================================
KV_PARSE = r'''
import json, os, re, signal, subprocess, threading, time, urllib.request

PORT = 8000
BASE = f"http://127.0.0.1:{PORT}"
FP16 = "Qwen/Qwen2.5-3B-Instruct"
KV_BYTES_PER_TOKEN = 36864

def clean(s):
    return re.sub(r"[^A-Za-z0-9一-鿿 :,.%x()=+\-\[\]_]", "", s)[:150]

LOGS = [("FP16", 0.5, "vllm_fp16_u0.5.log"), ("FP16", 0.7, "vllm_fp16_u0.7.log"),
        ("FP16", 0.9, "vllm_fp16_u0.9.log"), ("AWQ", 0.9, "vllm_awq_u0.9.log")]

def parse(log):
    t = open(log, errors="ignore").read()
    kv = None; conc = None
    for pat in (r"GPU KV cache size:\s*([\d,]+)\s*tokens",
                r"KV cache size:\s*([\d,]+)",
                r"kv[ _]cache[^\n]*?([\d,]{5,})\s*tokens",
                r"([\d,]{5,})\s*tokens of KV cache"):
        m = re.search(pat, t, re.I)
        if m: kv = int(m.group(1).replace(",", "")); break
    m = re.search(r"concurrency[^\d]*([\d.]+)x", t, re.I)
    if m: conc = float(m.group(1))
    if kv is None:
        print(f"  [{log}] 找不到 KV 池，含 KV/cache 的行如下（清理後）：")
        for l in t.splitlines():
            if re.search(r"KV|kv cache|GiB|concurren", l):
                print("   ", clean(l))
    return kv, conc

rows = []
print("=== 靜態：從留存的 log 重新解析 ===")
for model, util, log in LOGS:
    if not os.path.exists(log):
        print(f"  {log} 不存在"); rows.append({"model": model, "util": util}); continue
    kv, conc = parse(log)
    gb = round(kv * KV_BYTES_PER_TOKEN / 2**30, 2) if kv else None
    rows.append({"model": model, "util": util, "kv_pool_tokens": kv,
                 "kv_pool_gb_implied": gb, "max_concurrency_at_max_len": conc})
    print(f"  {model} util={util} | KV 池 {kv} tokens = {gb} GB | 8K 滿載並發 {conc}x")

fp = {r["util"]: r.get("kv_pool_tokens") for r in rows if r["model"] == "FP16"}
aw = next((r.get("kv_pool_tokens") for r in rows if r["model"] == "AWQ"), None)
p1 = ({"slope_tokens_per_0.2util_05_07": fp[0.7] - fp[0.5],
       "slope_tokens_per_0.2util_07_09": fp[0.9] - fp[0.7]}
      if all(fp.get(u) for u in (0.5, 0.7, 0.9)) else None)
p2 = (aw - fp[0.9]) if (aw and fp.get(0.9)) else None
print(f"P1 斜率：{p1}")
print(f"P2：AWQ 池 - FP16 池 = {p2} tokens = "
      f"{round(p2*KV_BYTES_PER_TOKEN/2**30,2) if p2 else '?'} GB（預測約 3.7 GB）")

# ---- P3：重啟一台 FP16@0.9，自動探測 metric 名再動態採樣 ----
FILLER = ("銀行法第十二條所稱擔保授信，謂對於銀行之授信，提供左列之一為擔保者："
          "不動產或動產抵押權、動產或權利質權、借款人營業交易所發生之應收票據、"
          "各級政府公庫主管機關、銀行或經政府核准設立之信用保證機構之保證。")
def make_prompt(n):
    return FILLER * max(1, n // len(FILLER)) + "\n\n根據以上內容，用一句話說明什麼是擔保授信。"

subprocess.run(["pkill", "-f", "vllm serve"]); time.sleep(10)
proc = subprocess.Popen(
    ["vllm", "serve", FP16, "--port", str(PORT), "--dtype", "float16",
     "--max-model-len", "8192", "--gpu-memory-utilization", "0.9"],
    stdout=open("vllm_p3.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid)
print("P3：啟動 server …")
for i in range(900):
    if proc.poll() is not None:
        print(open("vllm_p3.log").read()[-1500:]); raise SystemExit("P3 server 掛了")
    try:
        urllib.request.urlopen(f"{BASE}/v1/models", timeout=2); print("✓ 就緒"); break
    except Exception: time.sleep(2)

met = urllib.request.urlopen(f"{BASE}/metrics", timeout=5).read().decode()
names = sorted({m for m in re.findall(r"^([a-zA-Z_:][\w:]*)", met, re.M)
                if "cache" in m and ("usage" in m or "perc" in m)})
print("  metrics 候選：", [clean(n) for n in names])
METRIC = names[0] if names else None

def usage():
    if not METRIC: return None
    t = urllib.request.urlopen(f"{BASE}/metrics", timeout=5).read().decode()
    v = [float(x) for x in re.findall(re.escape(METRIC) + r"\S*\s+([\d.eE+-]+)", t)]
    return max(v) if v else None

def post(prompt, n):
    body = json.dumps({"model": FP16, "prompt": prompt, "max_tokens": n,
                       "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(f"{BASE}/v1/completions", data=body,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        for _ in r: pass

kv_pool = fp.get(0.9)
dyn = []
for target in (400, 1600, 4000):
    peak = {"v": 0.0}
    def watch():
        t0 = time.time()
        while time.time() - t0 < 30:
            u = usage()
            if u: peak["v"] = max(peak["v"], u)
            time.sleep(0.25)
    w = threading.Thread(target=watch, daemon=True); w.start()
    post(make_prompt(target), 64); time.sleep(0.6)
    used = round(peak["v"] * kv_pool) if (kv_pool and peak["v"]) else None
    dyn.append({"target_prompt_tokens": target, "peak_usage": round(peak["v"], 5),
                "implied_tokens_in_cache": used})
    print(f"  target={target:>5} | 峰值 usage {dyn[-1]['peak_usage']} | 折算 {used} tokens")
os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

out = {"meta": {"host": "Colab T4 16GB", "engine": "vLLM",
                "kv_bytes_per_token_theory": KV_BYTES_PER_TOKEN, "metric_used": METRIC,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
       "static": rows, "p1_slopes": p1, "p2_awq_minus_fp16_tokens": p2, "dynamic": dyn}
with open("vllm_kv_probe.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("✓ 已寫入 vllm_kv_probe.json（覆蓋第一輪全 None 的版本）")
'''

# ============================================================
# KV 探針第三輪：補 P1/P2。前兩輪的死因終於定位——不是 regex 錯，
# 是 **stdout 緩衝**：vLLM 輸出導到檔案時是全緩衝（非 tty），SIGTERM
# 一殺、還在緩衝裡的尾巴全丟。u0.9 與 AWQ 的 log 是 0 bytes、u0.5 只有
# 前 12KB；p3 那份活下來只因為它跑得久、輸出量大到多次 flush。
# 修法：子行程加 PYTHONUNBUFFERED=1，並在殺之前等 log 出現 KV 行。
#
# p3 的完整 log 已經把理論值驗掉一半：
#   GPU KV cache size: 187,824 tokens；Available KV cache memory: 6.45 GiB
#   187,824 × 36,864 bytes = 6.45 GiB ✓（36KB/token 成立，log 自己互證）
# 這一輪只補三格：FP16@0.5、FP16@0.7、AWQ@0.9（FP16@0.9 已有 187,824）。
# ============================================================
KV_FIX = r'''
import json, os, re, signal, subprocess, time, urllib.request

PORT = 8000
BASE = f"http://127.0.0.1:{PORT}"
FP16 = "Qwen/Qwen2.5-3B-Instruct"
AWQ  = "Qwen/Qwen2.5-3B-Instruct-AWQ"
KV_BYTES_PER_TOKEN = 36864
FP16_U09 = 187824   # 已由 vllm_p3.log 取得（完整 log 那一份）

ENV = dict(os.environ, PYTHONUNBUFFERED="1")

def grab(model, util, tag):
    log = f"vllm_{tag}.log"
    subprocess.run(["pkill", "-f", "vllm serve"]); time.sleep(10)
    proc = subprocess.Popen(
        ["vllm", "serve", model, "--port", str(PORT), "--dtype", "float16",
         "--max-model-len", "8192", "--gpu-memory-utilization", str(util)],
        stdout=open(log, "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=ENV)
    print(f"啟動 {model} util={util} …")
    kv = conc = None
    for i in range(900):
        if proc.poll() is not None:
            print(open(log, errors="ignore").read()[-1500:])
            raise SystemExit(f"{tag} 啟動即掛")
        t = open(log, errors="ignore").read()
        m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", t)
        if m:
            kv = int(m.group(1).replace(",", ""))
            c = re.search(r"Maximum concurrency for\s*[\d,]+\s*tokens per request:\s*([\d.]+)x", t)
            conc = float(c.group(1)) if c else None
            break
        time.sleep(2)
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    for _ in range(90):
        if proc.poll() is not None: break
        time.sleep(1)
    time.sleep(8)
    gb = round(kv * KV_BYTES_PER_TOKEN / 2**30, 2) if kv else None
    print(f"  {model} util={util} | KV 池 {kv} tokens = {gb} GB | 8K 滿載並發 {conc}x")
    return {"model": model.split("/")[-1], "util": util, "kv_pool_tokens": kv,
            "kv_pool_gb_implied": gb, "max_concurrency_at_max_len": conc}

rows = [grab(FP16, 0.5, "fix_fp16_u0.5"),
        grab(FP16, 0.7, "fix_fp16_u0.7"),
        {"model": "Qwen2.5-3B-Instruct", "util": 0.9, "kv_pool_tokens": FP16_U09,
         "kv_pool_gb_implied": round(FP16_U09 * KV_BYTES_PER_TOKEN / 2**30, 2),
         "max_concurrency_at_max_len": round(FP16_U09 / 8192, 2),
         "source": "vllm_p3.log（前一輪的完整 log）"},
        grab(AWQ, 0.9, "fix_awq_u0.9")]

fp = {r["util"]: r["kv_pool_tokens"] for r in rows if "AWQ" not in r["model"]}
aw = rows[-1]["kv_pool_tokens"]
p1 = {"0.5→0.7": fp[0.7] - fp[0.5] if fp.get(0.7) and fp.get(0.5) else None,
      "0.7→0.9": fp[0.9] - fp[0.7] if fp.get(0.9) and fp.get(0.7) else None}
p2 = aw - fp[0.9] if aw and fp.get(0.9) else None
print(f"\nP1 每 +0.2 util 的池增量：{p1}（預測約 8 萬 token）")
print(f"P2 AWQ − FP16（同 0.9）：{p2} tokens = "
      f"{round(p2 * KV_BYTES_PER_TOKEN / 2**30, 2) if p2 else '?'} GB（預測約 3.7 GB）")

out = {"meta": {"host": "Colab T4 16GB", "engine": "vLLM",
                "kv_bytes_per_token_theory": KV_BYTES_PER_TOKEN,
                "buffering_fix": "PYTHONUNBUFFERED=1＋殺行程前先等 log 出現 KV 行",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
       "static": rows, "p1_slopes": p1, "p2_awq_minus_fp16_tokens": p2}
with open("vllm_kv_fix.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("✓ 已寫入 vllm_kv_fix.json")
'''

if __name__ == "__main__":
    print(__doc__)
    print("=" * 60)
    print("Cell 1（安裝）：")
    print(SETUP)
    print("=" * 60)
    print("Cell 2（量測）：")
    print(BENCH)
    print("=" * 60)
    print("Cell 3（並發飽和點掃描）：")
    print(CONC_SWEEP)
    print("=" * 60)
    print("Cell 4（飽和點歸因探針：GPU 還是客戶端？）：")
    print(GPU_PROBE)
