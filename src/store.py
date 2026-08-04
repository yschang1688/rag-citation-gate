"""pgvector 連線與 embedding 呼叫。

embedding 走本機 Ollama（bge-m3，1024 維）——整條 RAG 不碰任何雲端 API，
成本為零、可離線重跑；代價是生成品質受限於本機模型，README 有明說。
"""
from __future__ import annotations

import os

import psycopg
import requests

PG = dict(
    host=os.environ.get("RAG_PG_HOST", "127.0.0.1"),
    port=int(os.environ.get("RAG_PG_PORT", "15432")),
    user=os.environ.get("RAG_PG_USER", "rag"),
    password=os.environ.get("RAG_PG_PASSWORD", "rag-dev-only"),
    dbname=os.environ.get("RAG_PG_DB", "rag"),
)
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = "bge-m3"
DIM = 1024

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS article (
    id        serial PRIMARY KEY,
    law       text NOT NULL,
    label     text NOT NULL,      -- 「第5條」
    text      text NOT NULL,
    embedding vector(%(dim)s) NOT NULL,
    UNIQUE (law, label)           -- 冪等 ingest 的錨點
);
""" % {"dim": DIM}


def connect():
    return psycopg.connect(**PG)


def embed(texts: list[str]) -> list[list[float]]:
    r = requests.post(f"{OLLAMA}/api/embed",
                      json={"model": EMBED_MODEL, "input": texts}, timeout=120)
    r.raise_for_status()
    out = r.json()["embeddings"]
    assert all(len(v) == DIM for v in out)
    return out


def to_vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"
