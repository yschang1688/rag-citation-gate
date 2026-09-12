"""檢索 → 生成 → 引文閘。

生成端走本機 Ollama（qwen3.5:9b）。模型被要求以 JSON 回覆並附引文，
但 pipeline 的正確性**不依賴模型聽話**——它不附引文、亂引、改寫引文，
一律被 citegate 擋成拒答。設計上模型是不可信元件，閘門才是可信元件。
"""
from __future__ import annotations

import json
import os
import re
import sys

import requests

from citegate import Citation, GateResult, verify
from store import OLLAMA, connect, embed, to_vec

GEN_MODEL = "qwen3.5:9b"
# 預設 5。可用 RAG_TOP_K 覆寫以便做 k 的對照實驗——調大 k 會提高召回，
# 但也把更多不相關條文送進生成端的 context，兩者要一起量才知道划不划算。
TOP_K = int(os.environ.get("RAG_TOP_K", "5"))

PROMPT = """你是法規問答助理。只能根據下面提供的條文回答，不得使用其他知識。

回答規則：
1. 以 JSON 回覆：{{"answer": "…", "citations": [{{"law": "法規名", "label": "第N條", "quote": "條文原文逐字節錄"}}]}}
2. quote 必須從提供的條文**逐字複製**，不得改寫、不得增刪字。
3. 只能引用下面提供的條文；若提供的條文不足以回答，answer 寫「依檢索到的條文無法回答」且 citations 為空陣列。

條文：
{context}

問題：{question}
"""


# 「第72-2條」「第5-1條」「第12條」都要抓得到。
ARTICLE_RE = re.compile(r"第\s*([0-9]+(?:[-－][0-9]+)?)\s*條")


def _cited_articles(question: str, k: int) -> list[dict]:
    """使用者指名條號時，直接精確匹配 label——這一路是確定性的，不經過向量空間。

    為什麼需要它（量測結論，不是直覺）：在 golden/questions-hard.json 的
    referential 組（六題指名條號），純稠密檢索只有 recall@5 = 0.667、top1 = 0.500；
    失敗長得都一樣——問「銀行法第72-2條」回第22條、問「洗錢防制法第10條」回第5條。
    **條號是符號不是語義**，「第10條」與「第5條」在向量空間裡極近，這是稠密表示法
    結構上解不了的問題，換更大的 embedding 模型也一樣。加上這一路之後
    referential 組是 recall 1.000 / MRR 1.000 / top1 1.000。

    另外量過但**沒有採用**的方案：加一路 BM25 做三路 RRF 融合。referential 確實
    也會好轉，但口語改寫組被 BM25 的噪聲拖垮（recall@5 0.917 → 0.500、top1 歸零）。
    詳見 bench/RETRIEVAL_FINDINGS.md。規則層沒有這個副作用，因為問句不含條號時
    它回空清單、自動退場，對其餘查詢是零影響。
    """
    nums = {n.replace("－", "-") for n in ARTICLE_RE.findall(question)}
    if not nums:
        return []
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT law, label, text FROM article")
        rows = cur.fetchall()
    out = []
    for law, label, text in rows:
        m = ARTICLE_RE.findall(label)
        if m and m[0].replace("－", "-") in nums:
            out.append(dict(law=law, label=label, text=text, dist=0.0, via="cited"))
    return out[:k]


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """條號直取 + 稠密檢索。前者命中時排在前面，其餘名額由向量檢索補滿。

    刻意不做的事：條號命中就短路回傳。同一條號可能存在於多部法規，而且使用者
    指名條號時往往還有語義成分（「第32條限制了哪一種授信」），保留向量那幾席
    讓語境有機會進來。
    """
    cited = _cited_articles(question, k)
    seen = {(c["law"], c["label"]) for c in cited}
    qv = embed([question])[0]
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT law, label, text, embedding <=> %s::vector AS dist
               FROM article ORDER BY dist LIMIT %s""", (to_vec(qv), k))
        dense = [dict(law=r[0], label=r[1], text=r[2], dist=float(r[3]), via="dense")
                 for r in cur.fetchall()]
    return (cited + [d for d in dense if (d["law"], d["label"]) not in seen])[:k]


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
