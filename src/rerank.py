"""cross-encoder 重排（bge-reranker-v2-m3）。預設關閉，`RERANK=cross` 才啟用。

為什麼是 opt-in 而不是預設開（量測結論，見 bench/RETRIEVAL_FINDINGS.md 實測二）：
colloquial 組五題改善、兩題退化（c07、c10 從第 1 名被擠到第 2），淨值為正但
不是白吃的午餐；穩態每題 1–4 秒（M1 MPS、pool=10），批次可接受、互動要斟酌。
這裡不做「什麼叫複雜查詢」的判斷器——那是保費大於理賠的東西。要壓平最壞情況，
把候選文字截到 MAX_CHARS 就夠：語料 p90 是 384 字，只有長尾會被截。

失敗處置：任何錯誤（套件沒裝、模型載不起來、分數數量對不上）一律回傳原順序，
並在 stderr 警告一次。重排是優化不是必要步驟，它壞掉時系統該退回沒有它的樣子。
"""
from __future__ import annotations

import os
import sys

MODE = os.environ.get("RERANK", "none")          # none | cross
MODEL_NAME = "BAAI/bge-reranker-v2-m3"
MAX_CHARS = int(os.environ.get("RERANK_MAX_CHARS", "400"))
POOL_MULT = 2                                    # 重排前召回 k×2 筆；池子要比 k 大才有東西可排

_MODEL = None
_WARNED = False


def enabled() -> bool:
    return MODE == "cross"


def _warn(msg: str) -> None:
    global _WARNED
    if not _WARNED:
        print(f"[rerank] {msg}；退回原順序", file=sys.stderr)
        _WARNED = True


def _load():
    """延後載入：只有真的要用重排時才吃這 2GB 記憶體。MPS 遇到未實作算子就退 CPU。"""
    global _MODEL
    if _MODEL is None:
        from FlagEmbedding import FlagReranker  # 需 torch，走 .venv-hybrid
        device = os.environ.get("RERANK_DEVICE", "mps")
        try:
            _MODEL = FlagReranker(MODEL_NAME, use_fp16=True, devices=device)
        except Exception:
            _MODEL = FlagReranker(MODEL_NAME, use_fp16=False, devices="cpu")
    return _MODEL


def cross_rerank(question: str, hits: list[dict], top_k: int) -> list[dict]:
    """對召回的候選逐一算 query-document 相關性分數，由高到低重排後截到 top_k。"""
    if len(hits) <= 1:
        return hits[:top_k]
    try:
        pairs = [[question, f"{h['law']}{h['label']} {(h.get('text') or '')[:MAX_CHARS]}"]
                 for h in hits]
        scores = _load().compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
        if len(scores) != len(hits):
            _warn(f"分數數量 {len(scores)} ≠ 候選數 {len(hits)}")
            return hits[:top_k]
    except Exception as e:  # noqa: BLE001
        _warn(f"{type(e).__name__}: {e}")
        return hits[:top_k]
    order = sorted(range(len(hits)), key=lambda i: -scores[i])
    out = []
    for i in order[:top_k]:
        h = dict(hits[i])
        h["rerank_score"] = round(float(scores[i]), 4)
        out.append(h)
    return out
