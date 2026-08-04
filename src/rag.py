"""檢索 → 生成 → 引文閘。

生成端走本機 Ollama（qwen3.5:9b）。模型被要求以 JSON 回覆並附引文，
但 pipeline 的正確性**不依賴模型聽話**——它不附引文、亂引、改寫引文，
一律被 citegate 擋成拒答。設計上模型是不可信元件，閘門才是可信元件。
"""
from __future__ import annotations

import json
import sys

import requests

from citegate import Citation, GateResult, verify
from store import OLLAMA, connect, embed, to_vec

GEN_MODEL = "qwen3.5:9b"
TOP_K = 5

PROMPT = """你是法規問答助理。只能根據下面提供的條文回答，不得使用其他知識。

回答規則：
1. 以 JSON 回覆：{{"answer": "…", "citations": [{{"law": "法規名", "label": "第N條", "quote": "條文原文逐字節錄"}}]}}
2. quote 必須從提供的條文**逐字複製**，不得改寫、不得增刪字。
3. 只能引用下面提供的條文；若提供的條文不足以回答，answer 寫「依檢索到的條文無法回答」且 citations 為空陣列。

條文：
{context}

問題：{question}
"""


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    qv = embed([question])[0]
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT law, label, text, embedding <=> %s::vector AS dist
               FROM article ORDER BY dist LIMIT %s""", (to_vec(qv), k))
        return [dict(law=r[0], label=r[1], text=r[2], dist=float(r[3]))
                for r in cur.fetchall()]


def generate(question: str, retrieved: list[dict]) -> dict:
    context = "\n\n".join(f"【{r['law']} {r['label']}】\n{r['text']}" for r in retrieved)
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": GEN_MODEL,
        "messages": [{"role": "user",
                      "content": PROMPT.format(context=context, question=question)}],
        "format": "json",
        "stream": False,
        # 關掉 thinking：實測同一題開 thinking 要 517 秒、關掉 1.5 秒，
        # 且本管線的正確性來自引文閘而非模型推理深度，思考鏈是純成本。
        "think": False,
        "options": {"temperature": 0},
    }, timeout=900)
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])


def ask(question: str) -> tuple[str, GateResult, list[dict]]:
    retrieved = retrieve(question)
    raw = generate(question, retrieved)
    citations = [Citation(c.get("law", ""), c.get("label", ""), c.get("quote", ""))
                 for c in raw.get("citations", [])]
    gate = verify(citations, retrieved)
    answer = raw.get("answer", "") if gate.ok else \
        "（已拒答）模型回答未通過引文驗證，不予呈現。"
    return answer, gate, retrieved


def main() -> None:
    question = " ".join(sys.argv[1:]) or "銀行對同一人的授信總餘額有什麼限制？"
    answer, gate, retrieved = ask(question)
    print(f"問：{question}\n")
    print(f"檢索 top-{len(retrieved)}：" +
          "、".join(f"{r['law']}{r['label']}" for r in retrieved) + "\n")
    print(f"答：{answer}\n")
    print(f"引文閘：{gate.verdict}")
    for c in gate.citations:
        print(f"  [{c.verdict}] {c.law} {c.label}：{c.quote[:40]}…")


if __name__ == "__main__":
    main()
