#!/usr/bin/env python3
"""
CONTINUOUS LIVE NEWS RAG (FINAL FIXED)
✔ Streaming-safe Pathway
✔ Deduplicated NewsAPI ingestion
✔ Persistent vector store
✔ Offline embeddings
"""

import os
import time
import csv
import json
import threading
import requests
import numpy as np
import pathway as pw
from pathlib import Path
from uuid import uuid4
from threading import Lock

# ======================================================
# CONFIG
# ======================================================

NEWS_API_KEY = os.getenv("NEWSAPI_KEY")
if not NEWS_API_KEY:
    raise RuntimeError("❌ NEWSAPI_KEY not set")

DATA_FILE = "live_news.csv"
VECTORS_FILE = "vectors.npy"
META_FILE = "metadata.json"
SEEN_FILE = "seen_urls.json"

EMBED_DIM = 64
POLL_INTERVAL = 15
lock = Lock()

# ======================================================
# STORAGE INIT
# ======================================================

def init_storage():
    if not Path(DATA_FILE).exists():
        with open(DATA_FILE, "w", newline="") as f:
            csv.writer(f).writerow(
                ["row_id", "article_id", "title", "content", "published_at"]
            )

    for file, default in [
        (META_FILE, {}),
        (SEEN_FILE, []),
    ]:
        if not Path(file).exists():
            with open(file, "w") as f:
                json.dump(default, f)

    if not Path(VECTORS_FILE).exists():
        np.save(VECTORS_FILE, np.empty((0, EMBED_DIM)))

    print("✅ Storage initialized")

init_storage()

# ======================================================
# OFFLINE EMBEDDINGS
# ======================================================

def embed_text(text: str) -> list[float]:
    vec = np.zeros(EMBED_DIM)
    for i, b in enumerate(text.encode("utf-8", errors="ignore")):
        vec[i % EMBED_DIM] += (b % 31) / 31.0
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm else vec.tolist()

# ======================================================
# NEWS POLLER
# ======================================================

def poll_newsapi():
    print("📡 NewsAPI poller started")

    while True:
        try:
            r = requests.get(
                "https://newsapi.org/v2/top-headlines",
                params={
                    "language": "en",
                    "pageSize": 10,
                    "apiKey": NEWS_API_KEY,
                },
                timeout=10,
            )
            data = r.json()
        except Exception as e:
            print("⚠️ API error:", e)
            time.sleep(POLL_INTERVAL)
            continue

        with lock:
            with open(SEEN_FILE) as f:
                seen = set(json.load(f))

            new_rows = []
            for a in data.get("articles", []):
                url = a.get("url")
                if not url or url in seen:
                    continue

                new_rows.append([
                    str(uuid4()),
                    url,
                    a.get("title", ""),
                    a.get("content") or a.get("description", ""),
                    a.get("publishedAt", ""),
                ])
                seen.add(url)

            if new_rows:
                with open(DATA_FILE, "a", newline="") as f:
                    csv.writer(f).writerows(new_rows)

                with open(SEEN_FILE, "w") as f:
                    json.dump(list(seen), f)

                print(f"📰 {len(new_rows)} new articles")
            else:
                print("⏳ No new articles")

        time.sleep(POLL_INTERVAL)

# ======================================================
# PATHWAY PIPELINE
# ======================================================

schema = pw.schema_builder(
    columns={
        "row_id": pw.column_definition(dtype=str, primary_key=True),
        "article_id": pw.column_definition(dtype=str),
        "title": pw.column_definition(dtype=str),
        "content": pw.column_definition(dtype=str),
        "published_at": pw.column_definition(dtype=str),
    }
)

news = pw.io.csv.read(
    DATA_FILE,
    schema=schema,
    mode="streaming",
)

docs = news.select(
    article_id=pw.this.article_id,
    text=pw.apply(lambda t, c: f"{t}. {c}", pw.this.title, pw.this.content),
)

@pw.udf
def embed_udf(text: str):
    return embed_text(text)

docs_vec = docs.with_columns(
    embedding=embed_udf(pw.this.text)
)

# ======================================================
# VECTOR PERSISTENCE
# ======================================================

def persist_vector(key, row, time, is_addition):
    if not is_addition:
        return

    with lock:
        with open(META_FILE) as f:
            meta = json.load(f)

        if row["article_id"] in meta:
            return

        vectors = np.load(VECTORS_FILE)
        vectors = np.vstack([vectors, np.array(row["embedding"])])

        meta[row["article_id"]] = row["text"]

        np.save(VECTORS_FILE, vectors)
        with open(META_FILE, "w") as f:
            json.dump(meta, f, indent=2)

        print("🧠 Vector stored")

pw.io.subscribe(docs_vec, persist_vector)

# ======================================================
# START PIPELINE (RUNS ON IMPORT)
# ======================================================

print("\n🚀 Continuous Live News RAG (FINAL)")
print("Pipeline running\n")

threading.Thread(target=poll_newsapi, daemon=True).start()

try:
    pw.run(background=True)
except KeyboardInterrupt:
    print("\n⏹ Stopped cleanly")
