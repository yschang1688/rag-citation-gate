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
