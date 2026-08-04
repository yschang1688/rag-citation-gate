"""基準集評估：分開量測檢索、生成、引文閘三層。

三層分開量是重點——RAG 出錯時「檢索沒撈到」與「模型亂編」要各自可歸因，
混在一個總分裡會讓改進方向失焦。

指標
----
retrieval_recall@k  可答題中，人工判定的正解條號有進 top-k 的比例
                    （閘門與生成都無法補救檢索的漏，所以它是天花板）
answered_rate       可答題中，通過引文閘、真的給出答案的比例
citation_precision  所有被輸出的引文中，逐字驗證為 VALID 的比例
false_answer_rate   **不可答題中，系統給出答案的比例——這是幻覺的直接量測，
                    目標為 0。** 拒答不是失敗，硬答才是。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag import GEN_MODEL, TOP_K, ask  # noqa: E402

GOLDEN = Path(__file__).resolve().parent.parent / "golden" / "questions.json"
OUT = Path(__file__).resolve().parent.parent / "golden" / "results.json"


def norm(s: str) -> str:
    return "".join(s.split())


def main() -> None:
    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))["cases"]
    rows = []
    t0 = time.time()
    for c in cases:
        answer, gate, retrieved = ask(c["q"])
        hit = None
        if c["answerable"]:
            hit = any(r["law"] == c["law"] and norm(r["label"]) == norm(c["expect_label"])
                      for r in retrieved)
        rows.append({
            "id": c["id"], "answerable": c["answerable"],
            "retrieval_hit": hit,
            "gate": gate.verdict,
            "citations": [{"law": x.law, "label": x.label, "verdict": x.verdict}
                          for x in gate.citations],
            "answered": gate.ok,
            "retrieved": [f"{r['law']}{r['label']}" for r in retrieved],
            "answer": answer,
        })
        print(f"{c['id']} {'可答' if c['answerable'] else '不可答'} "
              f"| 檢索命中={hit} | 閘={gate.verdict} | 給答={gate.ok}")

    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    cits = [c for r in rows for c in r["citations"]]
    m = {
        "model": GEN_MODEL, "top_k": TOP_K, "cases": len(rows),
        "retrieval_recall_at_k": round(sum(r["retrieval_hit"] for r in ans) / len(ans), 3),
        "answered_rate": round(sum(r["answered"] for r in ans) / len(ans), 3),
        "citation_precision": (round(sum(c["verdict"] == "VALID" for c in cits) / len(cits), 3)
                               if cits else None),
        "false_answer_rate": round(sum(r["answered"] for r in una) / len(una), 3),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print("\n" + json.dumps(m, ensure_ascii=False, indent=1))
    OUT.write_text(json.dumps({"metrics": m, "rows": rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
