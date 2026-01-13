#!/usr/bin/env python3
"""
STREAMLIT-COMPATIBLE NEWS FETCHER
Uses on-demand fetching instead of background threads
OPTIMIZED FOR: 1000 NewsAPI requests/day
"""

import os
import csv
import json
import requests
import numpy as np
from pathlib import Path
from uuid import uuid4
from threading import Lock
from datetime import datetime, timedelta
import time

# ======================================================
# CONFIG - OPTIMIZED FOR YOUR API LIMITS
# ======================================================

NEWS_API_KEY = os.getenv("NEWSAPI_KEY", "")
USE_LIVE_NEWS = bool(NEWS_API_KEY)

DATA_FILE = "live_news.csv"
VECTORS_FILE = "vectors.npy"
META_FILE = "metadata.json"
SEEN_FILE = "seen_urls.json"
API_STATS_FILE = "api_stats.json"

EMBED_DIM = 64
MAX_DAILY_REQUESTS = 1000
CATEGORIES = ["technology", "business", "politics"]
lock = Lock()

# ======================================================
# API USAGE TRACKER
# ======================================================

class APIUsageTracker:
    """Track API usage to stay within 1000 requests per day"""

    def __init__(self):
        self.load_stats()

    def load_stats(self):
        """Load API usage stats from file"""
        try:
            if os.path.exists(API_STATS_FILE):
                with open(API_STATS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except:
            pass
        return {}

    def save_stats(self, stats):
        """Save API usage stats to file"""
        try:
            with open(API_STATS_FILE, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
        except:
            pass

    def get_today_requests(self):
        """Get number of requests made today"""
        stats = self.load_stats()
        today = datetime.now().date().isoformat()
        return stats.get(today, {}).get("requests", 0)

    def increment_requests(self, count=1):
        """Increment request count for today"""
        stats = self.load_stats()
        today = datetime.now().date().isoformat()

        if today not in stats:
            stats[today] = {"requests": 0, "last_fetch": None}

        stats[today]["requests"] = stats[today].get("requests", 0) + count
        stats[today]["last_fetch"] = datetime.now().isoformat()
        self.save_stats(stats)

        return stats[today]["requests"]

    def can_make_request(self):
        """Check if we can make another request today"""
        today_requests = self.get_today_requests()
        return today_requests < MAX_DAILY_REQUESTS

    def get_last_fetch_time(self):
        """Get last fetch time"""
        stats = self.load_stats()
        today = datetime.now().date().isoformat()
        last_fetch = stats.get(today, {}).get("last_fetch")
        if last_fetch:
            try:
                return datetime.fromisoformat(last_fetch)
            except:
                pass
        return None

    def time_since_last_fetch(self):
        """Get time since last fetch in minutes"""
        last_fetch = self.get_last_fetch_time()
        if last_fetch:
            return (datetime.now() - last_fetch).total_seconds() / 60
        return 9999  # Very large number if never fetched

# Initialize API tracker
api_tracker = APIUsageTracker()

def get_newsapi_usage():
    """Get today's NewsAPI usage - for app.py"""
    return api_tracker.get_today_requests()

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

        # Create API stats file if it doesn't exist
        if not Path(API_STATS_FILE).exists():
            with open(API_STATS_FILE, "w", encoding="utf-8") as f:
                json.dump({}, f)
            print(f"✅ Created {API_STATS_FILE}")

        print("✅ Storage initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Storage initialization failed: {e}")
        return False

# ======================================================
# SMART NEWS FETCHER - ON-DEMAND
# ======================================================

def fetch_news_if_needed():
    """
    Smart news fetcher that runs on-demand
    Only fetches if:
    1. We have API key
    2. We're under daily limit
    3. It's been at least 30 minutes since last fetch
    4. We have few articles (< 50)
    """

    if not USE_LIVE_NEWS:
        print("ℹ️ No NewsAPI key, running with existing data")
        return False

    # Check daily limit
    today_requests = api_tracker.get_today_requests()
    if today_requests >= MAX_DAILY_REQUESTS:
        print(f"⛔ Daily API limit reached: {today_requests}/{MAX_DAILY_REQUESTS}")
        return False

    # Check time since last fetch (minimum 30 minutes between fetches)
    minutes_since_fetch = api_tracker.time_since_last_fetch()
    if minutes_since_fetch < 30:
        print(f"⏳ Too soon since last fetch: {minutes_since_fetch:.1f} minutes ago")
        return False

    # Check how many articles we already have
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_articles = list(reader)
            if len(existing_articles) > 100:
                print(f"ℹ️ Already have {len(existing_articles)} articles, skipping fetch")
                return False
    except:
        existing_articles = []

    print("📡 Fetching fresh news articles...")

    # Pick a category to fetch (rotate through categories)
    category_index = today_requests % len(CATEGORIES)
    category = CATEGORIES[category_index]

    try:
        # Fetch news from NewsAPI
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "language": "en",
            "pageSize": 10,  # Get 10 articles
            "apiKey": NEWS_API_KEY,
            "category": category
        }

        print(f"🌐 Requesting {category} news from NewsAPI...")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Increment API request count
        requests_made = api_tracker.increment_requests()
        print(f"📊 API Usage: {requests_made}/{MAX_DAILY_REQUESTS}")

        with lock:
            # Load seen URLs
            try:
                with open(SEEN_FILE, "r", encoding="utf-8") as f:
                    seen = set(json.load(f))
            except:
                seen = set()

            new_rows = []
            articles_added = 0

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
                articles_added += 1

            # Save new articles
            if new_rows:
                with open(DATA_FILE, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows(new_rows)

                # Update seen URLs
                with open(SEEN_FILE, "w", encoding="utf-8") as f:
                    json.dump(list(seen), f)

                print(f"✅ Added {articles_added} new {category} articles")
                if articles_added > 0:
                    # Show first article title
                    title = new_rows[0][2]
                    title_display = title[:60] + "..." if len(title) > 60 else title
                    print(f"   📰 Sample: {title_display}")

                # Update vectors with new articles
                update_vectors()

                return True
            else:
                print(f"ℹ️ No new {category} articles found")
                return False

    except requests.exceptions.RequestException as e:
        print(f"⚠️ API request error: {e}")
        return False
    except Exception as e:
        print(f"⚠️ Unexpected error fetching news: {e}")
        return False

# ======================================================
# VECTOR UPDATE FUNCTION
# ======================================================

def update_vectors():
    """Update vectors from new articles"""
    print("🔄 Updating vectors with new articles...")

    try:
        with lock:
            # Read all articles
            articles = []
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    articles.append(row)

            # Load existing metadata
            try:
                with open(META_FILE, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except:
                metadata = {}

            # Load existing vectors
            try:
                vectors = np.load(VECTORS_FILE)
            except:
                vectors = np.empty((0, EMBED_DIM))

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
                embedding = simple_embed_text(text)

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
                return True
            else:
                print("⏳ No new articles to vectorize")
                return False

    except Exception as e:
        print(f"⚠️ Error updating vectors: {e}")
        return False

# ======================================================
# SIMPLE EMBEDDING FUNCTION
# ======================================================

def simple_embed_text(text: str) -> list[float]:
    """Simple text embedding function"""
    if not text:
        text = ""

    # Convert to lowercase and clean
    text = text.lower()
    text = ''.join([c for c in text if c.isalnum() or c.isspace()])

    # Create simple embedding
    vec = np.zeros(EMBED_DIM)
    words = text.split()

    for word in words:
        # Simple hash-based position
        idx = abs(hash(word)) % EMBED_DIM
        vec[idx] += 1

    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec.tolist()

# ======================================================
# STREAMLIT-SAFE ENTRY POINT
# ======================================================

def start_background_rag():
    """
    Initialize the system - NO BACKGROUND THREADS
    Streamlit-safe version
    """
    print("🚀 Starting NewsFlow RAG System")
    print(f"📊 Daily NewsAPI Limit: {MAX_DAILY_REQUESTS} requests")
    print(f"🤖 Gemini AI Limit: 20 requests TOTAL")

    # Initialize storage
    if not init_storage():
        print("❌ Failed to initialize storage")
        return False

    print("✅ System initialized (no background threads)")
    return True

def check_and_fetch_news():
    """
    Check if we need to fetch news and do it if needed
    This should be called from app.py when appropriate
    """
    if not USE_LIVE_NEWS:
        return False

    # Always initialize storage first
    init_storage()

    # Try to fetch news
    return fetch_news_if_needed()

# For testing
if __name__ == "__main__":
    start_background_rag()
    success = fetch_news_if_needed()
    if success:
        print("✅ News fetch successful")
    else:
        print("ℹ️ No news fetched (check logs above)")
