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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# 2026-09-15 起實作搬到 src/rerank.py（生產路徑，RERANK=cross 啟用）；這裡只轉出口，
# 讓上面的 bench 指令與既有量測紀錄照舊可跑，不留第二份實作。
from rerank import MODEL_NAME, _load, cross_rerank  # noqa: E402,F401
