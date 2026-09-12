"""推理服務基準量測：在 Apple Silicon 上量 TTFT、decode 吞吐與上下文長度的關係。

為什麼是這個題目：面試高頻題「部署大模型服務真正關心什麼」的標準答案是
QPS／首 token 延遲／GPU 利用率／KV cache／量化精度。vLLM、SGLang、
TensorRT-LLM 這些引擎都要 CUDA，M1 跑不了——但那題問的是**現象與取捨**，
不是特定引擎的旗標。本腳本用 Ollama（llama.cpp + Metal）量出同一組現象：

  1. TTFT 隨 prompt 長度如何增長（prefill 是 compute-bound，理論上 O(n^2)）
  2. decode 速度是否與 prompt 長度無關（decode 是 memory-bound）
  3. 模型大小（4b vs 9b）對兩者的影響
  4. 並發如何換取吞吐、代價是延遲

用法：
    ./.venv/bin/python bench/serve_bench.py --out bench/results.json
    ./.venv/bin/python bench/serve_bench.py --quick        # 只跑 4b、兩個長度

量測方法：走 /api/generate 的 streaming 模式，**第一個 chunk 抵達的牆鐘時間**
就是 TTFT——不是用 Ollama 自報的 prompt_eval_duration，因為那不含排隊與
傳輸。自報值另外收下來做對照，兩者的差就是「引擎之外的開銷」。
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path

OLLAMA = "http://127.0.0.1:11434"
# 每個長度檔位用重複的中文語料墊出來；用中文是因為受測場景是中文法規 RAG，
# 英文 tokenizer 的 token/字元比不同，混用會讓長度軸失去意義。
FILLER = (
    "銀行法第十二條所稱擔保授信，謂對於銀行之授信，提供左列之一為擔保者："
    "不動產或動產抵押權、動產或權利質權、借款人營業交易所發生之應收票據、"
    "各級政府公庫主管機關、銀行或經政府核准設立之信用保證機構之保證。"
)
QUESTION = "\n\n根據以上內容，用一句話說明什麼是擔保授信。"


def _post_stream(model: str, prompt: str, num_predict: int, num_ctx: int):
    """回傳 (ttft_s, total_s, out_tokens, self_reported)。TTFT 以第一個內容 chunk 為準。

    兩個踩過的坑：
      - qwen3.5 是 thinking 模型，預設把 token 吐在 `thinking` 欄位、`response` 全空，
        照 `response` 計數會得到 decode = 0 tok/s。這裡以 `think: False` 關閉，
        讓量測對齊「一般問答服務」的行為。
      - 不帶 keep_alive 時模型會被卸載，下一輪的 TTFT 混進載入時間（實測 4.0s vs 0.34s，
        差一個數量級）。量的是服務穩態，不是冷啟動，所以釘住常駐。
    """
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": True,
        "think": False,
        "keep_alive": "10m",
        "options": {"num_predict": num_predict, "num_ctx": num_ctx, "temperature": 0},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=body,
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    n_out = 0
    final = {}
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            if not raw.strip():
                continue
            chunk = json.loads(raw)
            if chunk.get("response") or chunk.get("thinking"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n_out += 1
            if chunk.get("done"):
                final = chunk
    total = time.perf_counter() - t0
    self_reported = {
        # Ollama 以奈秒回報
        "prompt_eval_count": final.get("prompt_eval_count"),
        "prompt_eval_ms": (final.get("prompt_eval_duration") or 0) / 1e6,
        "eval_count": final.get("eval_count"),
        "eval_ms": (final.get("eval_duration") or 0) / 1e6,
        "load_ms": (final.get("load_duration") or 0) / 1e6,
    }
    return ttft or total, total, n_out, self_reported


def make_prompt(target_tokens: int) -> str:
    """用重複語料墊到目標 token 數附近。中文約 1 token ≈ 1 字，先粗估再由實測校正。"""
    if target_tokens <= 0:
        return QUESTION.strip()
    reps = max(1, target_tokens // len(FILLER))
    return (FILLER * reps) + QUESTION


def bench_latency(model: str, lengths: list[int], repeats: int, num_ctx: int) -> list[dict]:
    rows = []
    for target in lengths:
        prompt = make_prompt(target)
        # warm-up：第一次含模型載入，不計入
        _post_stream(model, prompt, 8, num_ctx)
        samples = []
        for _ in range(repeats):
            ttft, total, n_out, sr = _post_stream(model, prompt, 64, num_ctx)
            decode_tps = (n_out - 1) / (total - ttft) if total > ttft and n_out > 1 else 0.0
            samples.append((ttft, total, decode_tps, sr))
        ttfts = [s[0] for s in samples]
        tpss = [s[2] for s in samples]
        sr = samples[-1][3]
        rows.append({
            "model": model,
            "target_prompt_tokens": target,
            "actual_prompt_tokens": sr["prompt_eval_count"],
            "ttft_s_median": round(statistics.median(ttfts), 3),
            "ttft_s_min": round(min(ttfts), 3),
            "decode_tok_s_median": round(statistics.median(tpss), 2),
            # 短 prompt 的 prefill 速度會被固定開銷淹沒（分母太小），低於門檻不報，
            # 免得寫出「23 token 的 prefill 只有 200 tok/s」這種會誤導人的數字。
            "prefill_tok_s": (round((sr["prompt_eval_count"] or 0) / (sr["prompt_eval_ms"] / 1000), 1)
                              if sr["prompt_eval_ms"] and (sr["prompt_eval_count"] or 0) >= 200 else None),
            "self_reported_prompt_eval_ms": round(sr["prompt_eval_ms"], 1),
            "overhead_ms": round(statistics.median(ttfts) * 1000 - sr["prompt_eval_ms"], 1),
            "repeats": repeats,
        })
        print(f"  {model} ctx≈{sr['prompt_eval_count']:>5} tok | "
              f"TTFT {rows[-1]['ttft_s_median']:>6.3f}s | "
              f"decode {rows[-1]['decode_tok_s_median']:>5.2f} tok/s | "
              f"prefill {rows[-1]['prefill_tok_s'] or '—'} tok/s", flush=True)
    return rows


def bench_concurrency(model: str, levels: list[int], num_ctx: int) -> list[dict]:
    """並發：同時發 N 個請求，量總吞吐與單請求延遲的變化。"""
    prompt = make_prompt(400)
    _post_stream(model, prompt, 8, num_ctx)  # warm-up
    rows = []
    for n in levels:
        results: list[tuple] = []
        lock = threading.Lock()

        def worker():
            r = _post_stream(model, prompt, 48, num_ctx)
            with lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(n)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        total_out = sum(r[2] for r in results)
        rows.append({
            "model": model,
            "concurrency": n,
            "wall_s": round(wall, 2),
            "aggregate_tok_s": round(total_out / wall, 2),
            "ttft_s_median": round(statistics.median([r[0] for r in results]), 3),
            "ttft_s_max": round(max(r[0] for r in results), 3),
            "requests_per_min": round(n / wall * 60, 1),
        })
        print(f"  {model} concurrency={n:>2} | 總吞吐 {rows[-1]['aggregate_tok_s']:>6.2f} tok/s | "
              f"TTFT 中位 {rows[-1]['ttft_s_median']:>6.3f}s 最差 {rows[-1]['ttft_s_max']:>6.3f}s", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    models = ["qwen3.5:4b"] if args.quick else ["qwen3.5:4b", "qwen3.5:9b"]
    lengths = [0, 800] if args.quick else [0, 400, 1600, 4000]
    conc_levels = [1, 2] if args.quick else [1, 2, 4]

    out = {"meta": {
        "host": "Apple M1 16GB / macOS",
        "engine": "Ollama (llama.cpp + Metal)",
        "note": "TTFT 以第一個 streaming chunk 的牆鐘時間為準；self_reported 為 Ollama 自報的 prefill 時間，兩者差額為引擎外開銷。",
        "num_ctx": args.num_ctx,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, "latency": [], "concurrency": []}

    for m in models:
        print(f"\n=== 延遲 / 吞吐 vs 上下文長度：{m} ===", flush=True)
        out["latency"] += bench_latency(m, lengths, args.repeats, args.num_ctx)

    for m in models:
        print(f"\n=== 並發：{m} ===", flush=True)
        out["concurrency"] += bench_concurrency(m, conc_levels, args.num_ctx)

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✓ 已寫入 {p}")


if __name__ == "__main__":
    main()
