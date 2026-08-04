"""引文閘測試：重點不是乾淨輸入會過，是髒輸入擋在正確的判定。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from citegate import Citation, verify  # noqa: E402

RETRIEVED = [
    {"law": "銀行法", "label": "第2條",
     "text": "本法稱銀行，謂依本法組織登記，經營銀行業務之機構。"},
    {"law": "銀行法", "label": "第5-1條",
     "text": "本法稱收受存款，謂向不特定多數人收受款項或吸收資金，並約定返還本金或給付相當或高於本金之行為。"},
]


def one(law, label, quote):
    return verify([Citation(law, label, quote)], RETRIEVED)


def test_valid_verbatim_quote_passes():
    g = one("銀行法", "第2條", "謂依本法組織登記，經營銀行業務之機構")
    assert g.ok and g.citations[0].verdict == "VALID"


def test_whitespace_and_label_format_tolerated():
    # 條文原文有換行／模型引文帶空白、條號寫「第 2 條」——這些不是幻覺，不該擋
    g = verify([Citation("銀行法", "第 2 條", "謂依本法組織登記，\n經營銀行業務之機構")],
               RETRIEVED)
    assert g.ok


def test_paraphrase_rejected():
    # 意思對但改寫了字——正是最危險的一類，看起來像引文
    g = one("銀行法", "第2條", "銀行是依本法登記、經營銀行業務的機構")
    assert not g.ok and g.citations[0].verdict == "QUOTE_MISMATCH"


def test_unretrieved_article_rejected():
    # 第7條真的存在於銀行法，但不在這一輪檢索集合——憑記憶作答，擋
    g = one("銀行法", "第7條", "本法稱信託資金，謂銀行以受託人地位收受信託款項")
    assert not g.ok and g.citations[0].verdict == "UNRETRIEVED"


def test_short_quote_rejected():
    g = one("銀行法", "第2條", "銀行")
    assert not g.ok and g.citations[0].verdict == "QUOTE_TOO_SHORT"


def test_no_citation_is_not_pass():
    g = verify([], RETRIEVED)
    assert g.verdict == "NO_CITATION" and not g.ok


def test_one_bad_citation_rejects_whole_answer():
    g = verify([
        Citation("銀行法", "第2條", "謂依本法組織登記，經營銀行業務之機構"),
        Citation("銀行法", "第5-1條", "向特定多數人收受款項"),  # 「不特定」被改成「特定」
    ], RETRIEVED)
    assert g.verdict == "REJECTED"
    assert [c.verdict for c in g.citations] == ["VALID", "QUOTE_MISMATCH"]


def test_article_number_recovered_from_law_field():
    """實測發現的輸出格式：模型把條號寫進 law 欄、label 給「第1項」（項≠條）。

    這是格式容忍，不是驗證放寬——引文仍須逐字命中該條，見下面兩個探針。
    """
    g = verify([Citation("銀行法 第2條", "第1項",
                         "謂依本法組織登記，經營銀行業務之機構")], RETRIEVED)
    assert g.ok


def test_probe_wrong_article_with_verbatim_quote_still_rejected():
    """探針一：條號抽取不得退化成「隨便配一條有這段文字的」。

    引文逐字取自第5-1條，卻掛在第2條名下——若實作改成「在所有檢索結果裡
    找得到就算過」，本測試會翻綠，代表閘門被弄鬆了。
    """
    g = verify([Citation("銀行法 第2條", "第1項",
                         "向不特定多數人收受款項或吸收資金")], RETRIEVED)
    assert not g.ok and g.citations[0].verdict == "QUOTE_MISMATCH"


def test_probe_unparseable_label_rejected():
    """探針二：抽不出條號時必須擋，不得因「law 欄看起來對」就放行。"""
    g = verify([Citation("銀行法", "第1項",
                         "謂依本法組織登記，經營銀行業務之機構")], RETRIEVED)
    assert not g.ok and g.citations[0].verdict == "UNPARSEABLE"


def test_probe_gate_actually_gates():
    """探針自證：把逐字檢查退化成「開頭幾個字相符即過」應讓改寫測試失敗。

    這裡直接驗證「幾乎全對、只錯一字」的引文會被擋——若未來有人把
    比對邏輯改成前綴比對或相似度比對，本測試與 test_paraphrase_rejected
    會同時翻紅。
    """
    text = RETRIEVED[1]["text"]
    almost = text[:20] + "與" + text[21:]  # 換掉一個字
    g = one("銀行法", "第5-1條", almost)
    assert not g.ok
