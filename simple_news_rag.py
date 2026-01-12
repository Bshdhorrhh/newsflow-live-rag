#!/usr/bin/env python3
"""
CONTINUOUS LIVE NEWS RAG (STREAMLIT SAFE)
Simplified version for Streamlit Cloud compatibility
"""

import os
import time
import csv
import json
import threading
import requests
import numpy as np
from pathlib import Path
from uuid import uuid4
from threading import Lock

# ======================================================
# CONFIG
# ======================================================

# Use environment variable or default to mock mode
NEWS_API_KEY = os.getenv("NEWSAPI_KEY", "")
USE_LIVE_NEWS = bool(NEWS_API_KEY)

DATA_FILE = "live_news.csv"
VECTORS_FILE = "vectors.npy"
META_FILE = "metadata.json"
SEEN_FILE = "seen_urls.json"

EMBED_DIM = 64
POLL_INTERVAL = 60  # Reduced frequency for Streamlit Cloud
lock = Lock()

# ======================================================
# STORAGE INIT
# ======================================================

def init_storage():
    """Initialize storage files"""
    try:
        # Create data file if it doesn't exist
        if not Path(DATA_FILE).exists():
            with open(DATA_FILE, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["row_id", "article_id", "title", "content", "published_at"]
                )
            print(f"✅ Created {DATA_FILE}")

        # Create metadata file if it doesn't exist
        if not Path(META_FILE).exists():
            with open(META_FILE, "w", encoding="utf-8") as f:
                json.dump({}, f)
            print(f"✅ Created {META_FILE}")

        # Create seen URLs file if it doesn't exist
        if not Path(SEEN_FILE).exists():
            with open(SEEN_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
            print(f"✅ Created {SEEN_FILE}")

        # Create vectors file if it doesn't exist
        if not Path(VECTORS_FILE).exists():
            np.save(VECTORS_FILE, np.empty((0, EMBED_DIM)))
            print(f"✅ Created {VECTORS_FILE}")

        print("✅ Storage initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Storage initialization failed: {e}")
        return False

# ======================================================
# OFFLINE EMBEDDINGS
# ======================================================

def embed_text(text: str) -> list[float]:
    """Simple text embedding function"""
    vec = np.zeros(EMBED_DIM)
    text_bytes = text.encode("utf-8", errors="ignore")

    for i, b in enumerate(text_bytes):
        vec[i % EMBED_DIM] += (b % 31) / 31.0

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec.tolist()

# ======================================================
# NEWS POLLER (OPTIONAL)
# ======================================================

def poll_newsapi():
    """Poll NewsAPI for new articles"""
    if not USE_LIVE_NEWS:
        print("⚠️ NewsAPI key not set, running in mock mode")
        return

    print("📡 NewsAPI poller started")

    while True:
        try:
            # Fetch news from NewsAPI
            url = "https://newsapi.org/v2/top-headlines"
            params = {
                "language": "en",
                "pageSize": 5,  # Reduced for Streamlit Cloud
                "apiKey": NEWS_API_KEY,
                "category": "technology"  # Focus on tech news
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            with lock:
                # Load seen URLs
                with open(SEEN_FILE, "r", encoding="utf-8") as f:
                    seen = set(json.load(f))

                new_rows = []
                                # Debug print
                if new_rows:
                    print(f"📰 NEW ARTICLE ADDED:")
                    print(f"   Title: {article.get('title', 'No title')}")
                    print(f"   Source: {article.get('source', {}).get('name', 'Unknown')}")
                    print(f"   URL: {url}")
                for article in data.get("articles", []):
                    url = article.get("url")
                    if not url or url in seen:
                        continue

                    # Create new row
                    new_rows.append([
                        str(uuid4()),
                        url,
                        article.get("title", ""),
                        article.get("content") or article.get("description", ""),
                        article.get("publishedAt", "")
                    ])
                    seen.add(url)

                # Save new articles
                if new_rows:
                    with open(DATA_FILE, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerows(new_rows)

                    # Update seen URLs
                    with open(SEEN_FILE, "w", encoding="utf-8") as f:
                        json.dump(list(seen), f)

                    print(f"📰 Added {len(new_rows)} new articles")

        except requests.exceptions.RequestException as e:
            print(f"⚠️ API request error: {e}")
        except Exception as e:
            print(f"⚠️ Unexpected error in poller: {e}")

        # Wait before next poll
        time.sleep(POLL_INTERVAL)

# ======================================================
# VECTOR UPDATE FUNCTION
# ======================================================

def update_vectors():
    """Update vectors from new articles"""
    print("🔄 Checking for new articles to vectorize...")

    try:
        with lock:
            # Read all articles
            articles = []
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    articles.append(row)

            # Load existing metadata
            with open(META_FILE, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # Load existing vectors
            vectors = np.load(VECTORS_FILE)

            # Process new articles
            new_count = 0
            for article in articles:
                article_id = article["article_id"]

                # Skip if already in metadata
                if article_id in metadata:
                    continue

                # Create text for embedding
                text = f"{article['title']}. {article['content']}"

                # Generate embedding
                embedding = embed_text(text)

                # Update vectors
                vectors = np.vstack([vectors, np.array(embedding)])

                # Update metadata
                metadata[article_id] = {
                    "text": text,
                    "title": article["title"],
                    "published_at": article["published_at"]
                }

                new_count += 1

            # Save updated data
            if new_count > 0:
                np.save(VECTORS_FILE, vectors)
                with open(META_FILE, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

                print(f"🧠 Vectorized {new_count} new articles")
            else:
                print("⏳ No new articles to vectorize")

    except Exception as e:
        print(f"⚠️ Error updating vectors: {e}")

# ======================================================
# STREAMLIT-SAFE ENTRY POINT
# ======================================================

def start_background_rag():
    """Start the background RAG system"""
    print("🚀 Starting NewsFlow RAG System")

    # Initialize storage
    if not init_storage():
        print("❌ Failed to initialize storage, running in limited mode")
        return

    # Start news poller in a separate thread (if API key available)
    if USE_LIVE_NEWS:
        poller_thread = threading.Thread(target=poll_newsapi, daemon=True)
        poller_thread.start()
        print("✅ News poller started")
    else:
        print("ℹ️ Running without live news updates (no API key)")

    # Start vector updater in a separate thread
    def vector_updater():
        while True:
            update_vectors()
            time.sleep(POLL_INTERVAL * 2)  # Update less frequently

    updater_thread = threading.Thread(target=vector_updater, daemon=True)
    updater_thread.start()
    print("✅ Vector updater started")

    print("✅ NewsFlow RAG system is running")

# For direct execution
if __name__ == "__main__":
    start_background_rag()
