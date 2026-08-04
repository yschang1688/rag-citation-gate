"""語料 → embedding → pgvector。UPSERT 冪等：重跑筆數不變。"""
from __future__ import annotations

import json
from pathlib import Path

from store import SCHEMA, connect, embed, to_vec

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"
BATCH = 32


def main() -> None:
    files = sorted(CORPUS.glob("*.json"))
    if not files:
        raise SystemExit("data/corpus 為空——先跑 fetch_corpus.py")

    with connect() as conn:
        cur = conn.cursor()
        cur.execute(SCHEMA)
        total = 0
        for f in files:
            doc = json.loads(f.read_text(encoding="utf-8"))
            arts = doc["articles"]
            for i in range(0, len(arts), BATCH):
                chunk = arts[i:i + BATCH]
                # embedding 的輸入帶法規名與條號，讓「洗錢防制法第5條」這類
                # 指名道姓的查詢在向量空間也對得上
                vecs = embed([f"{doc['law']}{a['label']}\n{a['text']}" for a in chunk])
                for a, v in zip(chunk, vecs):
                    cur.execute(
                        """INSERT INTO article (law, label, text, embedding)
                           VALUES (%s, %s, %s, %s::vector)
                           ON CONFLICT (law, label) DO UPDATE
                           SET text = EXCLUDED.text, embedding = EXCLUDED.embedding""",
                        (doc["law"], a["label"], a["text"], to_vec(v)))
            total += len(arts)
            print(f"✓ {doc['law']}：{len(arts)} 條")
        cur.execute("SELECT COUNT(*) FROM article")
        print(f"庫內共 {cur.fetchone()[0]} 條（本輪處理 {total}）")


if __name__ == "__main__":
    main()
