"""幻覺注入演練：用**偽造的模型輸出**直接考閘門。

為什麼需要這支：基準集跑出 false_answer_rate = 0，但那 8 題不可答題是
**模型自己選擇拒答**的——閘門連上場的機會都沒有。「沒出事」不等於
「防線有效」，兩者必須分開證明。

這裡跳過模型，餵四種典型幻覺給閘門，看它擋不擋得住。檢索是真的（走
pgvector），只有生成端被替換成偽造輸出。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from citegate import Citation, verify  # noqa: E402
from rag import retrieve  # noqa: E402

QUESTION = "銀行可以經營哪些業務？"


def drill(retrieved: list[dict]) -> int:
    top = retrieved[0]
    other = next((r for r in retrieved[1:] if r["law"] == top["law"]), retrieved[1])
    quote = top["text"][:30]

    cases = [
        ("① 條號憑空捏造（庫裡根本沒有第9999條）",
         Citation(top["law"], "第9999條", quote), False),
        ("② 條號真實但這輪沒檢索到——憑訓練記憶作答",
         Citation("銀行法", "第129條", "違反第五十條第一項規定者"), False),
        ("③ 引文改寫：意思對、字不對（最像真引文的一種）",
         Citation(top["law"], top["label"], "銀行能夠經營的業務包括收受各類存款與辦理放款"), False),
        ("④ 張冠李戴：引文逐字為真，卻掛在別條名下",
         Citation(top["law"], top["label"], other["text"][:30]), False),
        ("⑤ 對照組：條號正確且引文逐字",
         Citation(top["law"], top["label"], quote), True),
    ]

    failures = 0
    print(f"問題：{QUESTION}")
    print("檢索：" + "、".join(f"{r['law']}{r['label']}" for r in retrieved) + "\n")
    for desc, cit, should_pass in cases:
        g = verify([cit], retrieved)
        ok = g.ok == should_pass
        failures += not ok
        print(f"{'✓' if ok else '✗'} {desc}\n"
              f"    閘門判定 {g.verdict}／{g.citations[0].verdict}"
              f"（預期{'放行' if should_pass else '擋下'}）")
    return failures


def main() -> int:
    retrieved = retrieve(QUESTION)
    failures = drill(retrieved)
    print(f"\n{'✓ 四種幻覺全數擋下，正常引文放行' if not failures else f'✗ {failures} 項未如預期'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
