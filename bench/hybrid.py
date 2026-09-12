"""三路檢索：條號規則 + BM25 詞彙 + bge-m3 稠密向量，以 RRF 融合。

## 為什麼是這三路

`bench/retrieval_eval.py --strategy dense` 在困難基準集上量到的失敗模式很明確：

  colloquial（口語改寫）  recall@5 0.917 / top1 0.583
  referential（指名條號） recall@5 0.667 / top1 0.500   ← 最差

referential 那組的失敗長得都一樣：問「銀行法第72-2條在限制什麼」，
top1 回第22條；問「洗錢防制法第10條」，top1 回第5條。**條號是符號不是語義**——
在向量空間裡「第10條」和「第5條」極近，因為它們語義上都是「某條法規」。
這不是模型不夠好，是稠密向量這個表示法結構上解不了的問題。

`src/ingest.py` 已經試圖用「把法規名與條號拼進被 embed 的文字」來緩解，
那個設計有註解說明意圖但從未被量測——量了才知道它沒有解決 referential 問題。

所以補的兩路都針對這個診斷，而不是因為「大家都說要 hybrid」：

  rule    從 query 抽出「第X條」直接精確匹配 label。零成本、可審計、命中即最高分。
  bm25    詞彙層檢索。中文用 character bigram 切詞（法規語料沒有空白分隔，
          且專有名詞多為 2–4 字，bigram 的召回表現對這類語料夠好且零依賴）。
  dense   現況那條，負責語義泛化——口語改寫題只有它接得住。

## 融合方式

RRF（Reciprocal Rank Fusion）：score = Σ weight / (k + rank)。選它而不是加權分數相加，
因為三路的分數量綱完全不同（cosine 距離 0–1、BM25 可以是 10 以上、規則是布林），
直接相加會讓量綱大的那路淹沒其他路——這正是「混合檢索的量綱陷阱」。
RRF 只吃名次不吃分數，天然免疫這個問題。

## 用法

    ./.venv/bin/python bench/retrieval_eval.py --strategy hybrid
    ./.venv/bin/python bench/retrieval_eval.py --strategy rule+dense
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from store import connect, embed, to_vec  # noqa: E402

# 「第72-2條」「第5-1條」「第12條」都要抓得到；全形數字不處理（語料與題目都用半形）
ARTICLE_RE = re.compile(r"第\s*([0-9]+(?:[-－][0-9]+)?)\s*條")
_CORPUS: list[dict] | None = None
_BM25: "BM25" | None = None


def bigrams(text: str) -> list[str]:
    """中文 character bigram。法規語料無空白分隔，且關鍵詞多為 2–4 字，
    bigram 能覆蓋「授信」「擔保」「洗錢」這類詞而不需要分詞器。
    英數字連續片段另外整段保留，避免把 'ATM' 拆成無意義的 bigram。"""
    toks = re.findall(r"[A-Za-z0-9]+", text)
    zh = re.sub(r"[^一-鿿]", "", text)
    toks += [zh[i:i + 2] for i in range(len(zh) - 1)]
    return toks


class BM25:
    """標準 BM25（k1=1.5, b=0.75）。語料只有 281 條，純 Python 完全夠用——
    這個規模引入 Elasticsearch 之類的依賴是保險料大於理賠率。"""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in docs) / self.N if self.N else 0.0
        self.tf = [Counter(d) for d in docs]
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        # +0.5 平滑，避免出現在多數文件的詞拿到負 idf
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = bigrams(query)
        scores = defaultdict(float)
        for t in q:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t)
                if not f:
                    continue
                dl = len(self.docs[i])
                scores[i] += idf * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return sorted(scores.items(), key=lambda x: -x[1])[:k]


def corpus() -> list[dict]:
    """一次性把 281 條讀進記憶體。資料量小，不值得為此設計增量索引。"""
    global _CORPUS, _BM25
    if _CORPUS is None:
        with connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT law, label, text FROM article ORDER BY id")
            _CORPUS = [dict(law=r[0], label=r[1], text=r[2]) for r in cur.fetchall()]
        # 與 ingest 一致：索引的文字帶上法規名與條號，讓詞彙層也能吃到條號
        _BM25 = BM25([bigrams(f"{c['law']}{c['label']} {c['text']}") for c in _CORPUS])
    return _CORPUS


def rule_search(question: str, k: int) -> list[int]:
    """條號精確匹配。query 說「第72-2條」就只回 label 對得上的那幾條
    （同一條號可能存在於多部法規，全部回傳，交給融合層與 dense 決定哪一部）。"""
    docs = corpus()
    nums = {n.replace("－", "-") for n in ARTICLE_RE.findall(question)}
    if not nums:
        return []
    out = []
    for i, c in enumerate(docs):
        m = ARTICLE_RE.findall(c["label"])
        if m and m[0].replace("－", "-") in nums:
            out.append(i)
    return out[:k]


def bm25_search(question: str, k: int) -> list[int]:
    corpus()
    return [i for i, _ in _BM25.search(question, k)]


def dense_search_idx(question: str, k: int) -> tuple[list[int], dict[int, float]]:
    """回傳 (corpus 索引, 索引→cosine 距離)。距離要留著：不可答題那組是靠
    top1 距離判斷「向量空間有沒有給出查無信號」，融合後若把距離抹成 0，
    那個分析就靜默失效了（第一版就是這樣，指標全變 0.0000 才發現）。"""
    docs = corpus()
    key = {(c["law"], c["label"]): i for i, c in enumerate(docs)}
    qv = embed([question])[0]
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT law, label, embedding <=> %s::vector AS dist
               FROM article ORDER BY dist LIMIT %s""", (to_vec(qv), k))
        rows = cur.fetchall()
    idx = [key[(r[0], r[1])] for r in rows if (r[0], r[1]) in key]
    dist = {key[(r[0], r[1])]: float(r[2]) for r in rows if (r[0], r[1]) in key}
    return idx, dist


def rrf(ranked: dict[str, list[int]], weights: dict[str, float], k: int,
        rrf_k: int = 60) -> list[int]:
    scores: defaultdict[int, float] = defaultdict(float)
    for name, ids in ranked.items():
        w = weights.get(name, 1.0)
        for rank, doc in enumerate(ids, 1):
            scores[doc] += w / (rrf_k + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]]


def search(question: str, k: int, strategy: str) -> list[dict]:
    docs = corpus()
    pool = max(k, 20)          # 融合前各路多召回一些，否則 RRF 沒有東西可以重排
    ranked: dict[str, list[int]] = {}
    dist: dict[int, float] = {}
    if "rule" in strategy:
        ranked["rule"] = rule_search(question, pool)
    if "bm25" in strategy or strategy == "hybrid":
        ranked["bm25"] = bm25_search(question, pool)
    if "dense" in strategy or strategy == "hybrid":
        ranked["dense"], dist = dense_search_idx(question, pool)
    # 規則命中是強信號（使用者指名了條號），權重高於另外兩路；
    # 但仍走融合而非直接短路——同一條號可能跨法規，語義那路要有機會排序。
    weights = {"rule": 3.0, "bm25": 1.0, "dense": 1.0}
    idx = rrf(ranked, weights, k) if len(ranked) > 1 else (
        next(iter(ranked.values()))[:k] if ranked else [])
    # dist 沿用 dense 那路的真實 cosine 距離；規則或 BM25 撈到但 dense 沒撈到的，
    # 標 nan 而不是 0——0 會被讀成「完全相同」，是最糟的預設值。
    # 帶上 text：重排層要看條文內容才能判斷相關性
    return [dict(law=docs[i]["law"], label=docs[i]["label"], text=docs[i]["text"],
                 dist=dist.get(i, float("nan"))) for i in idx]
