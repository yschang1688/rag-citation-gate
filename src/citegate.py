"""引文驗證閘（citation gate）——本 repo 的存在理由。

RAG 最常見的失效不是「檢索不到」，是**生成端看起來有憑有據、實際上沒有**：
引了不存在的條號、或把條文改寫幾個字再加上引號。防線不能建在
「請模型自我檢查」上——模型自述可信度本身就是待驗證的宣稱。

這裡的做法：模型輸出的每一條引文都要通過三道程式端檢查，
全部逐字、全部確定性、零 LLM 參與：

  1. RETRIEVED   引用的條文必須在**這一輪檢索到的集合**內——
                 引到庫裡有、但這輪沒檢索到的條文，等於憑訓練記憶作答，
                 一樣擋下（這是比「條號存在」更嚴的判準）。
  2. VERBATIM    引文去除空白後必須是該條全文的逐字子字串——改寫即擋。
  3. NONEMPTY    引文至少 8 個字——防「引一個字也算逐字」的空洞引用。

任何一條引文沒過，整個回答判 REJECTED，呈現給使用者的是拒答而非答案。
寧可拒答，不可背書。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MIN_QUOTE_CHARS = 8

# 條號抽取。模型常把條號寫進 law 欄（"銀行法 第32條"）而 label 給更細的
# 層級（"第1項"）——項是條之下的層級，不是條。這是**輸出格式的容忍**，
# 不是驗證的放寬：抽出條號後，VERBATIM 與 RETRIEVED 兩道檢查一字未改，
# 引文仍必須逐字命中**該條**的全文。抽不出條號一律擋（UNPARSEABLE）。
ARTICLE = re.compile(r"第\s*\d+(?:[-之]\d+)?\s*條")


def _norm(s: str) -> str:
    return "".join(s.split())


def _article_of(c: "Citation") -> str | None:
    for field in (c.label, c.law):
        m = ARTICLE.search(field or "")
        if m:
            return _norm(m.group())
    return None


def _law_of(c: "Citation") -> str:
    """law 欄可能夾帶條號（"銀行法 第32條"），取條號之前的法規名。"""
    return _norm(ARTICLE.sub("", c.law or ""))


@dataclass
class Citation:
    law: str
    label: str      # 「第5條」
    quote: str
    verdict: str = ""   # VALID / UNPARSEABLE / UNRETRIEVED / QUOTE_MISMATCH / QUOTE_TOO_SHORT


@dataclass
class GateResult:
    verdict: str            # PASS / REJECTED / NO_CITATION
    citations: list[Citation]

    @property
    def ok(self) -> bool:
        return self.verdict == "PASS"


def verify(citations: list[Citation], retrieved: list[dict]) -> GateResult:
    """retrieved: [{'law':…, 'label':…, 'text':…}, …]——只認這一輪檢索到的。"""
    if not citations:
        return GateResult("NO_CITATION", [])

    by_key = {(_norm(r["law"]), _norm(r["label"])): r["text"] for r in retrieved}
    all_ok = True
    for c in citations:
        art = _article_of(c)
        text = by_key.get((_law_of(c), art)) if art else None
        if art is None:
            c.verdict = "UNPARSEABLE"
        elif text is None:
            c.verdict = "UNRETRIEVED"
        elif len(_norm(c.quote)) < MIN_QUOTE_CHARS:
            c.verdict = "QUOTE_TOO_SHORT"
        elif _norm(c.quote) not in _norm(text):
            c.verdict = "QUOTE_MISMATCH"
        else:
            c.verdict = "VALID"
        all_ok = all_ok and c.verdict == "VALID"

    return GateResult("PASS" if all_ok else "REJECTED", citations)
