"""檢索層評估：在同一組題目上比較不同檢索策略，只量檢索、不跑生成。

為什麼把檢索層單獨拉出來評估：原本的 evaluate.py 是端到端的（檢索→生成→引文閘），
跑一輪 20 題要 12 分鐘，而且生成模型的隨機性會混進檢索的帳。要回答
「hybrid 到底值不值得加」，必須把檢索單獨量——每題不到一秒，可以反覆跑。

策略：
  dense     現況。bge-m3 稠密向量 + pgvector cosine，就是 src/rag.py 在用的那條。
  sparse    bge-m3 的詞彙權重（lexical weights），功能上接近 BM25。需 FlagEmbedding。
  hybrid    dense + sparse 以 RRF 融合。需 FlagEmbedding。
  rerank    hybrid 召回後再過 bge-reranker-v2-m3 重排。需 FlagEmbedding。

指標分 variant 報，不只報總平均——總體 recall 高不代表每一類都好，
這正是那些「總體 95%、關鍵意圖只有 70%」的案例會被平均數蓋掉的地方。

用法：
    ./.venv/bin/python bench/retrieval_eval.py --strategy dense
    ./.venv/bin/python bench/retrieval_eval.py --strategy dense --questions golden/questions.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from store import connect, embed, to_vec  # noqa: E402


def dense_search(question: str, k: int) -> list[dict]:
    qv = embed([question])[0]
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT law, label, text, embedding <=> %s::vector AS dist
               FROM article ORDER BY dist LIMIT %s""", (to_vec(qv), k))
        return [dict(law=r[0], label=r[1], text=r[2], dist=float(r[3]))
                for r in cur.fetchall()]


def rank_of(hits: list[dict], law: str | None, label: str | None) -> int | None:
    """正確條文在結果中的名次（1-based）；沒命中回 None。"""
    for i, h in enumerate(hits, 1):
        if h["label"] == label and (law is None or h["law"] == law):
            return i
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="golden/questions-hard.json")
    ap.add_argument("--strategy", default="dense",
                    choices=["dense", "bm25", "rule+dense", "bm25+dense", "hybrid"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rerank", default="none", choices=["none", "llm", "cross"])
    ap.add_argument("--pool", type=int, default=10,
                    help="重排前先召回幾筆。重排只能改順序、不能無中生有，"
                         "所以池子要比 k 大才有東西可排。")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # 有重排時先召回 pool 筆，重排後再截到 k；沒有重排就直接取 k。
    fetch_n = args.pool if args.rerank != "none" else args.k
    if args.strategy == "dense":
        base = lambda q: dense_search(q, fetch_n)  # noqa: E731
    else:
        from hybrid import search as hybrid_search  # 延後 import：dense 路徑不需要載語料
        base = lambda q: hybrid_search(q, fetch_n, args.strategy)  # noqa: E731

    if args.rerank == "llm":
        from rerank import llm_rerank
        searcher = lambda q: llm_rerank(q, base(q), args.k)  # noqa: E731
    elif args.rerank == "cross":
        from cross_rerank import cross_rerank  # 需要 torch，走 .venv-hybrid
        searcher = lambda q: cross_rerank(q, base(q), args.k)  # noqa: E731
    else:
        searcher = lambda q: base(q)[:args.k]  # noqa: E731

    cases = json.loads(Path(args.questions).read_text(encoding="utf-8"))["cases"]
    rows = []
    t0 = time.perf_counter()
    for c in cases:
        hits = searcher(c["q"])
        r = rank_of(hits, c.get("law"), c.get("expect_label")) if c["answerable"] else None
        rows.append({
            "id": c["id"], "variant": c.get("variant", "?"), "answerable": c["answerable"],
            "rank": r,
            "top1": f"{hits[0]['law']} {hits[0]['label']}" if hits else None,
            "top1_dist": round(hits[0]["dist"], 4) if hits else None,
            "expect": f"{c.get('law')} {c.get('expect_label')}" if c["answerable"] else None,
        })
    elapsed = time.perf_counter() - t0

    # 分組指標：可答題看 recall@k 與 MRR；不可答題沒有正解，改看 top1 距離的分布——
    # 距離若與可答題差不多，代表向量空間沒有給出「這題沒東西」的信號，
    # 拒答只能靠生成端，檢索端幫不上忙。這件事值得單獨知道。
    by = defaultdict(list)
    for r in rows:
        by[r["variant"]].append(r)

    report = {"strategy": args.strategy, "k": args.k, "questions": args.questions,
              "elapsed_s": round(elapsed, 1), "groups": {}, "rows": rows}
    print(f"\n策略 ={args.strategy}  k={args.k}  題庫={args.questions}  ({elapsed:.1f}s)\n")
    for variant, rs in by.items():
        ans = [r for r in rs if r["answerable"]]
        una = [r for r in rs if not r["answerable"]]
        if ans:
            hit = [r for r in ans if r["rank"]]
            recall = len(hit) / len(ans)
            mrr = statistics.mean([1 / r["rank"] for r in hit]) if hit else 0.0
            top1 = sum(1 for r in ans if r["rank"] == 1) / len(ans)
            report["groups"][variant] = {"n": len(ans), f"recall@{args.k}": round(recall, 3),
                                         "mrr": round(mrr, 3), "top1_acc": round(top1, 3)}
            print(f"  {variant:<13} n={len(ans):<3} recall@{args.k}={recall:.3f}  "
                  f"MRR={mrr:.3f}  top1={top1:.3f}")
            for r in ans:
                if r["rank"] != 1:
                    mark = "✗ 未召回" if not r["rank"] else f"→ 第 {r['rank']} 名"
                    print(f"      {r['id']} {mark}｜期望 {r['expect']}｜top1 {r['top1']} (d={r['top1_dist']})")
        # 可答與不可答分開報，不要因為同一組裡有可答題就吃掉不可答題的統計
        # （原 questions.json 沒有 variant 欄位，20 題全歸在同一組，曾讓 8 題不可答題靜默消失）
        if una:
            ds = [r["top1_dist"] for r in una]
            key = f"{variant}:unanswerable" if ans else variant
            report["groups"][key] = {"n": len(una), "top1_dist_median": round(statistics.median(ds), 4),
                                     "top1_dist_min": round(min(ds), 4)}
            print(f"  {key:<13} n={len(una):<3} top1 距離中位={statistics.median(ds):.4f} "
                  f"最近={min(ds):.4f}（無正解，看向量空間有無「查無」信號）")

    ansall = [r for r in rows if r["answerable"]]
    overall = sum(1 for r in ansall if r["rank"]) / len(ansall)
    report["overall_recall"] = round(overall, 3)
    print(f"\n  總體 recall@{args.k} = {overall:.3f}（{sum(1 for r in ansall if r['rank'])}/{len(ansall)}）"
          f" ← 分組看才有意義，這個數字只是對照用\n")

    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"✓ 已寫入 {args.out}")


if __name__ == "__main__":
    main()
