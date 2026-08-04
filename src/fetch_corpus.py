"""從全國法規資料庫抓取法規全文，以「條」為單位落成 JSON。

語料選法規不是隨機的：條號是天然的引文錨點——引文驗證需要一個
「可逐字比對的最小單位」，法條的條號＋條文正好是這個單位的教科書案例。
資料屬政府公開資訊（law.moj.gov.tw），僅作技術示範用途。
"""
from __future__ import annotations

import json
import re
import sys
import time
from html import unescape
from pathlib import Path

import requests
import truststore

# Python 3.14 內建 SSL 驗證會拒掉 law.moj.gov.tw 的憑證鏈
# （Missing Subject Key Identifier）；改走作業系統信任庫。
truststore.inject_into_ssl()

LAWS = {
    "G0380001": "銀行法",
    "I0050021": "個人資料保護法",
    "G0380131": "洗錢防制法",
}
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

# 先以列為單位切開、再逐列解析——單一大 regex 配 re.S 會在遇到
# 沒有條號錨點的列（刪除條文）時跨列吞噬，整段漏掉；逐列解析天然免疫。
ANCHOR = re.compile(r'name="(?P<flno>[\d-]+)">(?P<label>第[^<]+條)</a>')
# class 實際長相有兩種：`line-0000` 與 `line-0000 show-number`。
# 早期版本寫死 `line-\d+"`，於是帶 show-number 的行整段抓不到——
# 解析器不會報錯，只是少資料。這種靜默遺失比爆掉危險，故本檔以
# 「條數守衛」在來源端擋（見 parse_articles 回傳後的檢查）。
LINE = re.compile(r'<div class="line-\d+[^"]*">(.*?)</div>', re.S)
TAG = re.compile(r"<[^>]+>")
DELETED = re.compile(r"^[（(]刪除[)）]$")


def parse_articles(html: str) -> tuple[list[dict], int]:
    """回傳（條文清單, 頁面上的條號錨點總數）。後者供條數守衛比對。"""
    arts, anchors = [], 0
    for row in re.split(r'(?=<div class="row">)', html):
        m = ANCHOR.search(row)
        if not m:
            continue
        anchors += 1
        lines = [unescape(TAG.sub("", ln)).strip() for ln in LINE.findall(row)]
        text = "\n".join(l for l in lines if l)
        if not text or DELETED.match(text):
            continue  # 已刪除條文不進語料
        arts.append({
            "flno": m["flno"],
            "label": re.sub(r"\s+", "", m["label"]),  # 「第 5 條」→「第5條」
            "text": text,
        })
    return arts, anchors


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pcode, name in LAWS.items():
        url = f"https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode={pcode}"
        for attempt in range(3):
            try:
                r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == 2:
                    raise
                print(f"  {name} 第 {attempt+1} 次失敗（{type(e).__name__}），退避重試")
                time.sleep(10 * (attempt + 1))
        arts, anchors = parse_articles(r.text)
        # 條數守衛：頁面有幾個條號錨點，就該有幾條（扣掉已刪除條文）。
        # 少一條就停，不讓「解析器靜默漏資料」流進語料——RAG 的引文閘
        # 只能驗「引文是否忠於語料」，語料本身缺角它驗不出來。
        deleted = anchors - len(arts)
        if deleted > anchors * 0.2:
            sys.exit(f"{name}: {anchors} 個條號只解出 {len(arts)} 條，頁面結構可能已變")
        if len(arts) < 10:
            sys.exit(f"{name}: 只解析到 {len(arts)} 條，頁面結構可能已變")
        out = OUT_DIR / f"{pcode}.json"
        out.write_text(json.dumps(
            {"pcode": pcode, "law": name, "source": url,
             "fetched_at": time.strftime("%Y-%m-%d"), "articles": arts},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"✓ {name}：{len(arts)} 條（頁面 {anchors} 個條號，已刪除 {deleted}）→ {out.name}")
        time.sleep(2)  # 對公家站台客氣一點


if __name__ == "__main__":
    main()
