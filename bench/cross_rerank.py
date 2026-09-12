"""專用 cross-encoder 重排：bge-reranker-v2-m3。

## 這一路要回答的問題

`RETRIEVAL_FINDINGS.md` 留下一個沒有答案的問題：**專門訓練過的相關性判別，
能不能贏過向量相似度？** 前一輪用 qwen3.5:4b 做 listwise 重排的結果是零和
（c11 升第 1、c01 掉第 2，三個指標一字未動，耗時卻是 100 倍），
但那只證明「通用生成模型當評分者不夠」，沒有證明重排方向無效。

cross-encoder 與前者是兩種東西：它把 query 與候選**拼成一段**送進模型，
一次前向就吐出一個相關性分數。因為兩邊在同一個注意力視窗裡互相看得到，
理論上比「各自編碼成向量再算距離」更能捕捉細微的相關性差異——代價是
不能預先算好索引，只能對召回後的少量候選做。

## 為什麼選這個模型（2026-09 的地端選型）

bge-reranker-v2-m3：568M 參數、100+ 語言、2–4GB RAM，是 bge-m3 的官方配對。
在 M1 16GB 上跑得動，而且與現用的 embedding 同源、語言覆蓋一致。
另一個候選 Qwen3-Reranker 是 8B——同時還要留記憶體給 Ollama 的生成模型，
在 16GB 上不實際。實測環境：torch 2.14 + MPS。

## 用法

    ./.venv-hybrid/bin/python bench/retrieval_eval.py \
        --strategy rule+dense --rerank cross --pool 10
"""
from __future__ import annotations

import os

_MODEL = None
MODEL_NAME = "BAAI/bge-reranker-v2-m3"


def _load():
    """延後載入：只有真的要用重排時才吃這 2GB 記憶體。

    device 選 MPS（Apple GPU）；FlagEmbedding 在 MPS 上若遇到未實作的算子
    會拋錯，屆時退回 CPU——568M 的模型在 CPU 上也跑得動，只是慢。
    """
    global _MODEL
    if _MODEL is None:
        from FlagEmbedding import FlagReranker
        device = os.environ.get("RERANK_DEVICE", "mps")
        try:
            _MODEL = FlagReranker(MODEL_NAME, use_fp16=True, devices=device)
        except Exception:
            _MODEL = FlagReranker(MODEL_NAME, use_fp16=False, devices="cpu")
    return _MODEL


def cross_rerank(question: str, hits: list[dict], top_k: int) -> list[dict]:
    """對召回的候選逐一算 query-document 相關性分數，由高到低重排。

    與 LLM 版一樣的失敗處置：出任何問題就回傳原順序。重排是優化不是必要步驟，
    它壞掉時系統該退回沒有它的樣子。
    """
    if len(hits) <= 1:
        return hits[:top_k]
    try:
        pairs = [[question, f"{h['law']}{h['label']} {h.get('text') or ''}"] for h in hits]
        scores = _load().compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
        if len(scores) != len(hits):
            return hits[:top_k]
    except Exception:
        return hits[:top_k]
    order = sorted(range(len(hits)), key=lambda i: -scores[i])
    out = []
    for i in order[:top_k]:
        h = dict(hits[i])
        h["rerank_score"] = round(float(scores[i]), 4)
        out.append(h)
    return out
