"""重排層：把召回回來的候選重新排序。

## 為什麼做重排而不是繼續加召回路數

`RETRIEVAL_FINDINGS.md` 量到的瓶頸位置很明確——採用條號規則層之後：

    colloquial   recall@5 = 0.917   top1 = 0.583    ← 差距 0.334
    referential  recall@5 = 1.000   top1 = 1.000    ← 已無空間

**recall 與 top1 的差距，就是重排能拿走的部分。** 逐題看：c02 排第 5、c08 排第 3、
c11 排第 2、c12 排第 3——正確答案都已經在候選裡，只是順序不對。這是排序問題，
不是召回問題；再加 sparse 或 hybrid 對這四題無能為力（而且 BM25 那次已證明
詞彙訊號在口語查詢上是噪聲）。

## 兩種 reranker，這裡先做便宜的那個

- `llm`（本檔實作）：用本機既有的 qwen3.5:4b 做 listwise 重排。零安裝、
  現在就能跑。代價是慢（每題一次生成）且不是專用模型。
- `cross-encoder`（待接）：bge-reranker-v2-m3，568M 參數、2–4GB RAM，
  是 bge-m3 的官方配對，在 M1 16GB 上跑得動。需要 torch + FlagEmbedding。

**兩者不是同一種東西，不要混為一談**：cross-encoder 是專為「query-document 相關性」
訓練的判別模型，一次前向就出分數；LLM reranker 是拿通用生成模型當評分者，
更貴、更慢，但不需要額外權重。先做 LLM 版是為了**先確認重排這個方向真的有收益**，
再決定值不值得為專用模型付出安裝與記憶體成本——順序反過來就是先買保險再看要不要理賠。

用法：
    ./.venv/bin/python bench/retrieval_eval.py --strategy rule+dense --rerank llm
"""
from __future__ import annotations

import json
import re

import requests

OLLAMA = "http://127.0.0.1:11434"
RERANK_MODEL = "qwen3.5:4b"     # 用小模型：重排是排序任務，不需要 9b 的生成能力

PROMPT = """你是法規檢索的重排器。下面是使用者問題與若干候選條文。

請判斷每則候選條文**能否直接回答該問題**，並由最相關到最不相關重新排序。
只看內容相關性，不要考慮條號大小或出現順序。

以 JSON 回覆，格式：{{"order": [候選編號由最相關到最不相關]}}
必須包含且只包含下列所有編號，不得新增或遺漏。

問題：{question}

候選：
{candidates}
"""


def llm_rerank(question: str, hits: list[dict], top_k: int) -> list[dict]:
    """listwise 重排。選 listwise 而非逐一打分（pointwise）的原因：
    pointwise 要 N 次呼叫，listwise 一次；而且模型能看到候選之間的相對關係，
    在「哪個更相關」這種判斷上比各自獨立打分穩。

    失敗一律回傳原順序——重排是**優化**不是**必要步驟**，它壞掉時系統應該
    退回沒有它的樣子，而不是拒絕服務。這裡的失敗包含：模型回非法 JSON、
    回的編號有缺漏或超出範圍、呼叫逾時。
    """
    if len(hits) <= 1:
        return hits[:top_k]
    listing = "\n".join(
        f"[{i}] {h['law']}{h['label']}：{(h.get('text') or '')[:160]}"
        for i, h in enumerate(hits))
    try:
        r = requests.post(f"{OLLAMA}/api/chat", json={
            "model": RERANK_MODEL,
            "messages": [{"role": "user",
                          "content": PROMPT.format(question=question, candidates=listing)}],
            "format": "json", "stream": False, "think": False,
            "keep_alive": "10m",
            "options": {"temperature": 0},
        }, timeout=180)
        r.raise_for_status()
        order = json.loads(r.json()["message"]["content"]).get("order", [])
    except Exception:
        return hits[:top_k]

    # 驗證：必須是原集合的一個排列。少一個、多一個、超出範圍都視為失敗。
    # 不做「盡量修補」——修補會把模型的部分幻覺當成有效輸出，
    # 而重排失敗的正確處置是退回原順序，不是勉強用一個殘缺的排序。
    if not isinstance(order, list) or sorted(order) != list(range(len(hits))):
        return hits[:top_k]
    return [hits[i] for i in order][:top_k]


def parse_rank_hint(text: str) -> int | None:
    """從模型輸出裡挖出第一個整數，給 pointwise 模式備用（目前未啟用）。"""
    m = re.search(r"-?\d+", text or "")
    return int(m.group()) if m else None
