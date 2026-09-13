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
