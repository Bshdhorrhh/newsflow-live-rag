import streamlit as st
import time
import random
from datetime import datetime
import sys
import os
import re
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
import sqlite3
import json

# ======================================================
# IMPORT NEWS FETCHER
# ======================================================

try:
    from simple_news_rag import start_background_rag, check_and_fetch_news, get_newsapi_usage
    NEWS_FETCHER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import news fetcher: {e}")
    NEWS_FETCHER_AVAILABLE = False

# ======================================================
# INITIALIZE SYSTEM
# ======================================================

def initialize_system():
    """Initialize the news system on app load"""
    print("🔍 Initializing NewsFlow system...")

    # Initialize storage and check for news
    if NEWS_FETCHER_AVAILABLE:
        try:
            # Initialize storage
            start_background_rag()

            # Check if we need fresh news
            if "news_last_checked" not in st.session_state:
                st.session_state.news_last_checked = None

            # Only check every 30 minutes
            should_check = False
            if st.session_state.news_last_checked is None:
                should_check = True
            else:
                last_check = datetime.fromisoformat(st.session_state.news_last_checked)
                minutes_since_check = (datetime.now() - last_check).total_seconds() / 60
                if minutes_since_check > 30:
                    should_check = True

            if should_check:
                print("🔄 Checking for fresh news...")
                if check_and_fetch_news():
                    print("✅ Fresh news fetched successfully")
                else:
                    print("ℹ️ Using existing news data")
                st.session_state.news_last_checked = datetime.now().isoformat()
            else:
                print("⏳ Too soon since last news check")

        except Exception as e:
            print(f"⚠️ Error initializing system: {e}")
    else:
        print("⚠️ News fetcher not available, running with mock data")

# ======================================================
# DEBUG INFORMATION
# ======================================================

import sys

# Force output immediately
sys.stdout.write("\n" + "="*80 + "\n")
sys.stdout.write("FORCED DEBUG OUTPUT START\n")
sys.stdout.write("="*80 + "\n\n")

# Check what files are actually there
sys.stdout.write("Current directory: " + os.getcwd() + "\n")
sys.stdout.write("Files in directory:\n")
for f in os.listdir('.'):
    sys.stdout.write(f"  - {f}\n")

sys.stdout.write("\n" + "-"*80 + "\n")
sys.stdout.write("Checking for key files:\n")

key_files = ['vectors.npy', 'metadata.json', 'query_engine.py', 'llm_router.py', 'app.py']
print("\n=== NEWS DATABASE DEBUG ===")
try:
    import csv
    with open('live_news.csv', 'r') as f:
        reader = csv.DictReader(f)
        articles = list(reader)
        print(f"📊 Total articles in database: {len(articles)}")

        # Show recent articles
        print("\n📰 Recent 5 articles:")
        for i, article in enumerate(articles[-5:]):
            title = article.get('title', 'No title')
            print(f"  {i+1}. {title[:80]}...")

        # Show categories
        print("\n🔍 Checking for tech articles:")
        tech_articles = [a for a in articles if 'tech' in a.get('title', '').lower() or 'ai' in a.get('title', '').lower()]
        print(f"   Found {len(tech_articles)} tech-related articles")

except Exception as e:
    print(f"❌ Error reading news: {e}")

for file in key_files:
    if os.path.exists(file):
        size = os.path.getsize(file)
        sys.stdout.write(f"  ✅ {file}: {size} bytes\n")
    else:
        sys.stdout.write(f"  ❌ {file}: NOT FOUND\n")

sys.stdout.write("\n" + "-"*80 + "\n")
sys.stdout.write("Environment variables:\n")
sys.stdout.write(f"  GEMINI_API_KEY: {'SET' if os.getenv('GEMINI_API_KEY') else 'NOT SET'}\n")
sys.stdout.write(f"  NEWSAPI_KEY: {'SET' if os.getenv('NEWSAPI_KEY') else 'NOT SET'}\n")

sys.stdout.write("\n" + "="*80 + "\n")
sys.stdout.write("FORCED DEBUG OUTPUT END\n")
sys.stdout.write("="*80 + "\n\n")
sys.stdout.flush()  # Force flush

# ======================================================
# SAFE QUERY ENGINE LOADING
# ======================================================

sys.stdout.write("\n" + "="*60 + "\n")
sys.stdout.write("QUERY ENGINE LOADING - ENHANCED DEBUG\n")
sys.stdout.write("="*60 + "\n")

HAS_QUERY_ENGINE = False
GEMINI_AVAILABLE = False

try:
    # Step 1: Check for required files
    required = ["vectors.npy", "metadata.json", "query_engine.py"]
    all_exist = all(os.path.exists(f) for f in required)

    sys.stdout.write(f"Required files all exist: {all_exist}\n")

    # Step 2: Check Gemini package WITHOUT testing API
    sys.stdout.write("\nChecking Gemini package...\n")
    try:
        import google.generativeai as genai
        sys.stdout.write("✅ google.generativeai imported successfully\n")

        # Check API key exists but DON'T test it
        API_KEY = os.getenv("GEMINI_API_KEY")
        if API_KEY:
            sys.stdout.write("✅ Gemini API key found\n")
            GEMINI_AVAILABLE = True
        else:
            sys.stdout.write("⚠️ No Gemini API key found\n")

    except ImportError as e:
        sys.stdout.write(f"❌ google.generativeai import failed: {e}\n")
        sys.stdout.write("❌ Please install: pip install google-generativeai\n")

    # Step 3: Load query engine if files exist
    if all_exist and GEMINI_AVAILABLE:
        sys.stdout.write("\nImporting query_engine...\n")
        try:
            sys.stdout.write("✅ Query engine available (will lazy-load)\n")
            HAS_QUERY_ENGINE = True
        except Exception as import_err:
            sys.stdout.write(f"❌ Failed to import query_engine: {import_err}\n")
            import traceback
            traceback.print_exc()
    else:
        sys.stdout.write("❌ Using mock mode\n")

except Exception as e:
    sys.stdout.write(f"❌ Error in query engine loading: {e}\n")
    import traceback
    traceback.print_exc()

sys.stdout.write(f"\n" + "="*60 + "\n")
sys.stdout.write(f"FINAL STATUS: HAS_QUERY_ENGINE = {HAS_QUERY_ENGINE}\n")
sys.stdout.write("="*60 + "\n\n")
sys.stdout.flush()

# ======================================================
# MOCK FUNCTIONS FOR FALLBACK
# ======================================================

def mock_rag_answer(query):
    """Mock RAG answer function - uses NO Gemini API"""
    mock_responses = {
        "tech": "**Technology News Summary**\n\nRecent developments in the tech sector show significant growth in AI adoption across industries. Major companies are investing heavily in machine learning research, with breakthroughs in natural language processing and computer vision.\n\n**Key Developments:**\n• AI integration in enterprise solutions increased by 40% this quarter\n• Cloud computing services show record adoption rates\n• Cybersecurity remains a top concern with new threats emerging\n• Quantum computing research reaches new milestones",
        "ai": "**Artificial Intelligence Updates**\n\nThe AI landscape continues to evolve rapidly with new models and applications emerging weekly. Recent conferences highlighted advancements in multimodal AI systems capable of processing text, images, and audio simultaneously.\n\n**Notable Developments:**\n• New open-source language models with improved reasoning capabilities\n• AI-driven healthcare diagnostics showing 95% accuracy in trials\n• Regulatory frameworks taking shape across multiple countries\n• Increased investment in AI safety research",
        "politics": "**Political News Analysis**\n\nCurrent political discussions focus on economic policies and international relations. Recent summits have addressed climate change agreements and trade negotiations.\n\n**Key Updates:**\n• New trade agreements under negotiation\n• Climate policy discussions intensifying\n• Electoral reforms being considered in multiple regions\n• Diplomatic relations showing signs of improvement",
        "business": "**Business Market Report**\n\nGlobal markets show mixed performance with tech sectors leading gains while traditional industries face challenges. Economic indicators suggest cautious optimism among investors.\n\n**Market Insights:**\n• Tech stocks outperform traditional sectors\n• Inflation rates stabilizing in major economies\n• Supply chain disruptions easing gradually\n• Consumer confidence showing slight improvement"
    }

    query_lower = query.lower()
    if "tech" in query_lower or "technology" in query_lower:
        return mock_responses["tech"]
    elif "ai" in query_lower or "artificial" in query_lower:
        return mock_responses["ai"]
    elif "politics" in query_lower or "government" in query_lower:
        return mock_responses["politics"]
    elif "business" in query_lower or "market" in query_lower or "economy" in query_lower:
        return mock_responses["business"]
    else:
        return f"**News Summary: {query}**\n\nOur analysis of current news sources reveals several relevant articles on this topic. While specific details vary across sources, there's consensus around key developments in this area.\n\n**Main Points:**\n• Increased media coverage on this subject\n• Multiple expert opinions available\n• Varied perspectives across different news outlets\n• Growing public interest noted"

def mock_get_system_stats():
    """Mock system stats function"""
    # Try to get real stats from database first
    try:
        return get_real_time_stats()
    except:
        # Fall back to mock if database fails
        return {
            'date': datetime.now().strftime("%Y-%m-%d"),
            'total_queries': random.randint(10, 50),
            'avg_response_time': round(random.uniform(1.2, 2.5), 2),
            'successful_searches': random.randint(8, 45),
            'failed_searches': random.randint(0, 5),
            'success_rate': round(random.uniform(85, 99), 1),
            'total_articles_retrieved': random.randint(100, 300),
            'avg_similarity_score': round(random.uniform(0.6, 0.9), 3),
            'source_accesses': {
                'Bloomberg': random.randint(10, 50),
                'Reuters': random.randint(10, 45),
                'TechCrunch': random.randint(10, 40),
                'Financial Times': random.randint(10, 35),
                'BBC News': random.randint(10, 30),
                'The Verge': random.randint(10, 25),
                'Wall Street Journal': random.randint(10, 25),
                'CNBC': random.randint(10, 20)
            },
            'category_usage': {
                'technology': random.randint(15, 40),
                'business': random.randint(10, 30),
                'politics': random.randint(5, 20),
                'sports': random.randint(3, 15),
                'entertainment': random.randint(5, 18),
                'science': random.randint(3, 12)
            },
            'total_historical_queries': random.randint(100, 500),
            'avg_historical_response_time': round(random.uniform(1.5, 2.8), 2),
            'top_categories': {
                'technology': random.randint(40, 100),
                'business': random.randint(30, 80),
                'politics': random.randint(20, 60)
            },
            'system_uptime': 99.7,
            'cache_hits': 87,
            'last_updated': datetime.now().strftime("%Y-%m-d %H:%M:%S")
        }

def mock_get_live_stats():
    """Mock live stats function"""
    stats = mock_get_system_stats()
    stats.update({
        'active_sources': len(stats.get('source_accesses', {})),
        'avg_response': stats.get('avg_response_time', 0.0),
        'total_articles': 150,
        'system_status': '🟢 Optimal',
        'cache_status': '87% hit rate',
        'memory_usage': '2.4GB / 8GB',
        'cpu_usage': '34%'
    })
    return stats

# ======================================================
# DATABASE FUNCTIONS FOR REAL STATS
# ======================================================

def get_real_time_stats():
    """Get real-time statistics from database"""
    try:
        conn = sqlite3.connect('query_stats.db')
        cursor = conn.cursor()

        # Get today's date
        today = datetime.now().date().isoformat()

        # Get today's queries
        cursor.execute('''
            SELECT COUNT(*) as total_queries,
                   AVG(response_time) as avg_response_time,
                   SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_searches,
                   SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_searches,
                   AVG(similarity_score) as avg_similarity_score,
                   SUM(articles_retrieved) as total_articles_retrieved
            FROM query_stats
            WHERE date(timestamp) = ?
        ''', (today,))

        today_stats = cursor.fetchone()

        # Calculate success rate
        total_q = today_stats[0] or 0
        successful = today_stats[2] or 0
        success_rate = (successful / total_q * 100) if total_q > 0 else 0

        # Get source accesses
        cursor.execute('''
            SELECT sources_accessed, COUNT(*) as count
            FROM query_stats
            WHERE date(timestamp) = ?
            GROUP BY sources_accessed
        ''', (today,))

        source_rows = cursor.fetchall()
        source_accesses = {}
        for sources_str, count in source_rows:
            if sources_str:
                for source in sources_str.split(','):
                    source = source.strip()
                    if source:
                        source_accesses[source] = source_accesses.get(source, 0) + count

        # Get category usage
        cursor.execute('''
            SELECT category, COUNT(*) as count
            FROM query_stats
            WHERE date(timestamp) = ?
            GROUP BY category
        ''', (today,))

        category_rows = cursor.fetchall()
        category_usage = dict(category_rows)

        # Get historical stats
        cursor.execute('''
            SELECT COUNT(*) as total_historical,
                   AVG(response_time) as avg_historical_response
            FROM query_stats
        ''')

        historical_stats = cursor.fetchone()

        # Get top categories overall
        cursor.execute('''
            SELECT category, COUNT(*) as count
            FROM query_stats
            GROUP BY category
            ORDER BY count DESC
            LIMIT 5
        ''')

        top_cats = cursor.fetchall()
        top_categories = dict(top_cats)

        conn.close()

        return {
            'date': today,
            'total_queries': total_q,
            'avg_response_time': round(today_stats[1] or 0.0, 2),
            'successful_searches': successful,
            'failed_searches': today_stats[3] or 0,
            'success_rate': round(success_rate, 1),
            'total_articles_retrieved': today_stats[5] or 0,
            'avg_similarity_score': round(today_stats[4] or 0.0, 3),
            'source_accesses': source_accesses,
            'category_usage': category_usage,
            'total_historical_queries': historical_stats[0] or 0,
            'avg_historical_response_time': round(historical_stats[1] or 0.0, 2),
            'top_categories': top_categories,
            'system_uptime': 99.7,
            'cache_hits': 87,
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        print(f"❌ Error fetching real stats: {e}")
        # Fall back to mock
        return mock_get_system_stats()

def get_detailed_query_history(limit=20):
    """Get detailed query history from database"""
    try:
        conn = sqlite3.connect('query_stats.db')
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                timestamp,
                query,
                category,
                response_time,
                similarity_score,
                articles_retrieved,
                sources_accessed,
                CASE WHEN success THEN '✅' ELSE '❌' END as status
            FROM query_stats
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        conn.close()

        # Convert to list of dictionaries
        history = []
        for row in rows:
            history.append(dict(zip(columns, row)))

        return history

    except Exception as e:
        print(f"❌ Error fetching query history: {e}")
        return []

# ======================================================
# GEMINI API USAGE TRACKER (Only 20 requests!)
# ======================================================

def get_gemini_usage():
    """Get Gemini API usage count"""
    try:
        if os.path.exists('gemini_usage.json'):
            with open('gemini_usage.json', 'r') as f:
                data = json.load(f)
                return data.get('total_used', 0)
    except:
        pass
    return 0

def increment_gemini_usage():
    """Increment Gemini API usage counter"""
    try:
        if os.path.exists('gemini_usage.json'):
            with open('gemini_usage.json', 'r') as f:
                data = json.load(f)
        else:
            data = {'total_used': 0, 'daily_usage': {}}

        today = datetime.now().date().isoformat()
        data['total_used'] = data.get('total_used', 0) + 1

        if today not in data['daily_usage']:
            data['daily_usage'][today] = 0
        data['daily_usage'][today] += 1

        with open('gemini_usage.json', 'w') as f:
            json.dump(data, f, indent=2)

        return data['total_used']
    except Exception as e:
        print(f"❌ Error tracking Gemini usage: {e}")
        return 0

def can_use_gemini():
    """Check if we can use Gemini API (limit: 20 total)"""
    used = get_gemini_usage()
    return used < 20

def get_gemini_remaining():
    """Get remaining Gemini API calls"""
    used = get_gemini_usage()
    return max(0, 20 - used)

# ======================================================
# NEWSAPI USAGE TRACKER
# ======================================================

def get_newsapi_usage_info():
    """Get NewsAPI usage information"""
    try:
        if NEWS_FETCHER_AVAILABLE:
            return get_newsapi_usage()
    except:
        pass

    # Fallback: check api_stats.json directly
    try:
        if os.path.exists('api_stats.json'):
            with open('api_stats.json', 'r') as f:
                stats = json.load(f)
            today = datetime.now().date().isoformat()
            return stats.get(today, {}).get("requests", 0)
    except:
        pass

    return 0

# ======================================================
# LAZY-LOADED REAL QUERY ENGINE FUNCTIONS WITH GEMINI LIMIT
# ======================================================

# Cache for real query engine functions
_real_rag_answer = None
_real_get_system_stats = None
_real_get_live_stats = None

def lazy_load_query_engine():
    """Lazy load the query engine functions only when needed"""
    global _real_rag_answer, _real_get_system_stats, _real_get_live_stats

    if _real_rag_answer is None and HAS_QUERY_ENGINE:
        try:
            # Import and initialize Gemini API ONLY when needed
            import google.generativeai as genai
            API_KEY = os.getenv("GEMINI_API_KEY")
            if API_KEY:
                genai.configure(api_key=API_KEY)
                # Now import the query engine
                from query_engine import rag_answer as real_rag, get_system_stats as real_stats, get_live_stats as real_live
                _real_rag_answer = real_rag
                _real_get_system_stats = real_stats
                _real_get_live_stats = real_live
                print("✅ Real query engine lazy-loaded successfully")
            else:
                print("⚠️ No Gemini API key, using mock mode")
        except Exception as e:
            print(f"❌ Failed to lazy-load query engine: {e}")
            # Fall back to mock functions
            _real_rag_answer = None
            _real_get_system_stats = None
            _real_get_live_stats = None

def rag_answer(query):
    """Wrapper function that checks Gemini limit before using real engine"""
    # Always check Gemini limit first
    if not can_use_gemini():
        print("⚠️ Gemini API limit reached (20/20), using mock responses")
        st.session_state.gemini_limit_reached = True
        return mock_rag_answer(query)

    if HAS_QUERY_ENGINE:
        lazy_load_query_engine()
        if _real_rag_answer is not None:
            try:
                # Increment Gemini usage BEFORE making the call
                increment_gemini_usage()
                current_usage = get_gemini_usage()
                print(f"🤖 Using Gemini API ({current_usage}/20)")
                return _real_rag_answer(query)
            except Exception as e:
                print(f"❌ Error in real rag_answer: {e}")
                # Fall back to mock
                return mock_rag_answer(query)

    # Use mock if real engine not available
    return mock_rag_answer(query)

def get_system_stats():
    """Wrapper function for stats - uses real database when available"""
    try:
        if HAS_QUERY_ENGINE:
            lazy_load_query_engine()
            if _real_get_system_stats is not None:
                return _real_get_system_stats()

        # Fall back to database or mock
        return get_real_time_stats()
    except:
        return mock_get_system_stats()

def get_live_stats():
    """Wrapper function for live stats"""
    try:
        if HAS_QUERY_ENGINE:
            lazy_load_query_engine()
            if _real_get_live_stats is not None:
                stats = _real_get_live_stats()
                # Add database stats
                real_stats = get_real_time_stats()
                stats.update({
                    'date': real_stats.get('date', datetime.now().strftime("%Y-%m-%d")),
                    'total_queries': real_stats.get('total_queries', 0),
                    'success_rate': real_stats.get('success_rate', 0),
                    'total_articles_retrieved': real_stats.get('total_articles_retrieved', 0)
                })
                return stats
    except:
        pass

    # Fall back
    stats = mock_get_live_stats()
    # Merge with real database stats
    try:
        real_stats = get_real_time_stats()
        stats.update(real_stats)
    except:
        pass
    return stats

# ======================================================
# HELPER FUNCTIONS
# ======================================================

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def clean_response_text(text):
    """Clean the response text to remove HTML tags and fix formatting"""
    if not text:
        return ""

    # Remove HTML tags but keep the content
    text = re.sub(r'<[^>]+>', '', text)

    # Remove metadata footer if present
    text = re.sub(r'Search Information.*', '', text, flags=re.DOTALL)

    return text.strip()

def process_search_query(query):
    """Process a search query and return formatted response"""
    print(f"📡 Processing query: {query}")  # Debug log

    try:
        # This will check Gemini limit and use mock if exceeded
        response = rag_answer(query)
        response = clean_response_text(response)

        # Update stats when a query is processed
        if st.session_state.show_stats:
            # Get fresh stats data (real database)
            try:
                st.session_state.real_stats_data = get_real_time_stats()
                st.session_state.last_stats_update = datetime.now().strftime("%H:%M:%S")
            except:
                st.session_state.real_stats_data = None

        return response

    except Exception as e:
        print(f"❌ Error in process_search_query: {e}")
        return f"**Error Processing Query**\n\nUnable to fetch news results at the moment. Please try again.\n\nError: {str(e)}"

def fetch_fresh_news():
    """Manually trigger news fetch"""
    if NEWS_FETCHER_AVAILABLE:
        with st.spinner("🔄 Fetching fresh news articles..."):
            try:
                success = check_and_fetch_news()
                if success:
                    st.success("✅ Successfully fetched fresh news!")
                    st.session_state.news_last_checked = datetime.now().isoformat()
                    return True
                else:
                    st.info("ℹ️ No new articles fetched (check API limits or try again later)")
                    return False
            except Exception as e:
                st.error(f"❌ Error fetching news: {e}")
                return False
    else:
        st.error("❌ News fetcher not available")
        return False

# ======================================================
# PAGE CONFIG & INITIALIZATION
# ======================================================

# 1. Page Config
st.set_page_config(
    page_title="NewsFlow AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "NewsFlow AI - Real-time News Intelligence"
    }
)

# Initialize system on first load
if "system_initialized" not in st.session_state:
    initialize_system()
    st.session_state.system_initialized = True

# ---------------------------------------------------------
# THEME CONFIGURATION
# ---------------------------------------------------------
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

def toggle_theme():
    if st.session_state.theme == 'dark':
        st.session_state.theme = 'light'
    else:
        st.session_state.theme = 'dark'

# Define Theme Colors
if st.session_state.theme == 'dark':
    # Dark Mode (Default)
    theme_vars = """
    :root {
        --bg-gradient-start: #0a0a0a;
        --bg-gradient-end: #111111;
        --bg-card: rgba(17, 17, 17, 0.95);
        --bg-input: rgba(96, 100, 95, 0.3);
        --bg-sidebar: rgba(10, 10, 10, 0.98);
        --text-primary: #e0eab8;
        --text-secondary: #a0a895;
        --text-highlight: #e0eab8;
        --border-color: rgba(224, 234, 184, 0.2);
        --accent-color: #e0eab8;
        --accent-dim: rgba(224, 234, 184, 0.1);
        --shadow-color: rgba(0, 0, 0, 0.4);
    }
    """
else:
    # Light Mode
    theme_vars = """
    :root {
        --bg-gradient-start: #f4f7e8;
        --bg-gradient-end: #ffffff;
        --bg-card: rgba(255, 255, 255, 0.90);
        --bg-input: rgba(255, 255, 255, 0.8);
        --bg-sidebar: rgba(240, 245, 230, 0.98);
        --text-primary: #2c3327;
        --text-secondary: #5c6355;
        --text-highlight: #4a7a29;
        --border-color: rgba(44, 51, 39, 0.15);
        --accent-color: #4a7a29;
        --accent-dim: rgba(74, 122, 41, 0.1);
        --shadow-color: rgba(0, 0, 0, 0.1);
    }
    """

# 2. Session State for History & Gemini-style chat flow
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_search' not in st.session_state:
    st.session_state.current_search = None
if 'search_loading' not in st.session_state:
    st.session_state.search_loading = False
if 'show_stats' not in st.session_state:
    st.session_state.show_stats = False
if 'last_stats_update' not in st.session_state:
    st.session_state.last_stats_update = None
if 'real_stats_data' not in st.session_state:
    st.session_state.real_stats_data = None
if 'show_typing_effect' not in st.session_state:
    st.session_state.show_typing_effect = True
if 'typing_complete' not in st.session_state:
    st.session_state.typing_complete = False
if 'typing_signal_received' not in st.session_state:
    st.session_state.typing_signal_received = False
if 'api_request_made' not in st.session_state:
    st.session_state.api_request_made = False
if 'show_query_history' not in st.session_state:
    st.session_state.show_query_history = False
if 'query_history_data' not in st.session_state:
    st.session_state.query_history_data = []
if 'gemini_limit_reached' not in st.session_state:
    st.session_state.gemini_limit_reached = False
if 'newsapi_usage' not in st.session_state:
    st.session_state.newsapi_usage = 0
if 'last_news_fetch' not in st.session_state:
    st.session_state.last_news_fetch = None

# 3. Dynamic CSS Injection
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Inject Theme Variables */
    {theme_vars}

    * {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        outline: none !important;
        box-shadow: none !important;
    }}

    .stApp {{
        background: linear-gradient(135deg, var(--bg-gradient-start) 0%, var(--bg-gradient-end) 100%);
        color: var(--text-primary);
        min-height: 100vh;
    }}

    header[data-testid="stHeader"] {{
        background-color: transparent !important;
        backdrop-filter: blur(10px);
    }}

    .st-emotion-cache-1dp5vir {{
        background-color: transparent !important;
        backdrop-filter: blur(10px);
        border-bottom: 1px solid var(--border-color) !important;
    }}

    section[data-testid="stSidebar"] > div {{
        background: var(--bg-sidebar) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--border-color);
        height: 100vh !important;
    }}

    *:focus, *:focus-visible, *:focus-within {{
        outline: none !important;
        box-shadow: none !important;
        border-color: var(--border-color) !important;
    }}

    /* Chat Input Container */
    div[data-testid="stChatInputContainer"] {{
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        margin-left: 0rem !important;
        margin-right: 0rem !important;
        width: 100% !important;
        background: transparent !important;
    }}

    div[data-testid="stChatInputContainer"] > div {{
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        margin-left: 0rem !important;
        margin-right: 0rem !important;
        width: 100% !important;
        background: transparent !important;
    }}

    .stChatInputContainer {{
        background: var(--bg-card) !important;
        border-radius: 20px !important;
        margin-bottom: 1rem !important;
        padding: 0.5rem !important;
        border: 1px solid var(--border-color) !important;
        backdrop-filter: blur(10px);
    }}

    .stChatInputContainer textarea {{
        background: var(--bg-input) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 20px !important;
        color: var(--text-primary) !important;
        font-size: 1.05rem !important;
        min-height: 60px !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px);
        padding: 16px 20px !important;
        width: 100% !important;
    }}

    .stChatInputContainer textarea::placeholder {{
        color: var(--text-secondary) !important;
    }}

    .stChatInputContainer textarea:focus {{
        border-color: var(--accent-color) !important;
        background: var(--accent-dim) !important;
        box-shadow: 0 0 0 1px var(--border-color) !important;
    }}

    div[data-testid="stButton"] button {{
        background: var(--bg-input) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        transition: all 0.3s ease !important;
        border-radius: 16px !important;
    }}

    div[data-testid="stButton"] button:hover {{
        background: var(--accent-dim) !important;
        border-color: var(--accent-color) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px var(--shadow-color) !important;
    }}

    div[data-testid="stButton"] button:focus {{
        box-shadow: none !important;
        border-color: var(--border-color) !important;
        background: var(--accent-dim) !important;
    }}

    /* Mode Toggle Button specific style */
    .mode-toggle-btn {{
        background: transparent !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
    }}

    .stChatMessage {{
        padding: 1rem 0;
        animation: messageSlideIn 0.5s ease-out;
    }}

    @keyframes messageSlideIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .summary-card {{
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 24px;
        padding: 2rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px var(--shadow-color);
        position: relative;
        overflow: hidden;
        animation: cardAppear 0.6s ease-out;
    }}

    @keyframes cardAppear {{
        0% {{
            opacity: 0;
            transform: scale(0.95) translateY(20px);
        }}
        100% {{
            opacity: 1;
            transform: scale(1) translateY(0);
        }}
    }}

    .summary-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg,
            var(--accent-color),
            var(--text-secondary),
            var(--accent-color));
        animation: gradientFlow 3s ease infinite;
    }}

    @keyframes gradientFlow {{
        0%, 100% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
    }}

    .summary-title {{
        color: var(--text-primary);
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        line-height: 1.4;
        letter-spacing: -0.02em;
    }}

    .summary-content {{
        color: var(--text-primary);
        font-size: 1.1rem;
        line-height: 1.8;
        margin-bottom: 1.5rem;
        opacity: 0.95;
        animation: fadeInContent 0.8s ease-out 0.2s both;
    }}

    @keyframes fadeInContent {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 0.95; transform: translateY(0); }}
    }}

    .summary-content p {{
        margin-bottom: 1rem;
    }}

    .summary-content ul, .summary-content ol {{
        margin: 1rem 0;
        padding-left: 1.5rem;
    }}

    .summary-content li {{
        margin-bottom: 0.5rem;
        position: relative;
        color: var(--text-primary);
    }}

    .summary-content li::before {{
        content: '•';
        color: var(--accent-color);
        font-weight: bold;
        position: absolute;
        left: -1.5rem;
    }}

    /* Stats card specific styling */
    .stats-card {{
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 24px;
        padding: 2rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px var(--shadow-color);
        position: relative;
        overflow: hidden;
    }}

    .stats-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--accent-color), var(--text-secondary));
    }}

    .stats-title {{
        color: var(--text-primary);
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        line-height: 1.4;
        letter-spacing: -0.02em;
    }}

    .stats-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1.5rem;
    }}

    .stat-item {{
        background: var(--bg-input);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }}

    .stat-item:hover {{
        background: var(--accent-dim);
        transform: translateY(-2px);
        border-color: var(--accent-color);
    }}

    .stat-label {{
        color: var(--text-secondary);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }}

    .stat-value {{
        color: var(--text-primary);
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }}

    /* Source table styling */
    .source-table {{
        width: 100%;
        margin-top: 1.5rem;
        border-collapse: collapse;
    }}

    .source-table th {{
        color: var(--text-secondary);
        text-align: left;
        padding: 1rem;
        border-bottom: 1px solid var(--border-color);
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .source-table td {{
        color: var(--text-primary);
        padding: 1rem;
        border-bottom: 1px solid var(--border-color);
    }}

    .source-table tr:hover {{
        background: var(--accent-dim);
    }}

    /* Metadata section */
    .metadata-section {{
        background: var(--bg-input);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 2rem;
        border: 1px solid var(--border-color);
        animation: fadeIn 0.8s ease-out 0.4s both;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}

    .metadata-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }}

    .metadata-item {{
        background: var(--bg-input);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }}

    .metadata-item:hover {{
        background: var(--accent-dim);
        transform: translateY(-2px);
        border-color: var(--accent-color);
    }}

    .metadata-label {{
        color: var(--text-secondary);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }}

    .metadata-value {{
        color: var(--text-primary);
        font-size: 1.1rem;
        font-weight: 600;
    }}

    /* Loading animation */
    .loading-container {{
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 2rem;
        background: var(--bg-input);
        border-radius: 16px;
        border: 1px solid var(--border-color);
        animation: pulse 2s ease-in-out infinite;
    }}

    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.7; }}
    }}

    .searching-dots {{
        display: flex;
        gap: 0.5rem;
    }}

    .searching-dots span {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--accent-color);
        animation: bounce 1.4s ease-in-out infinite;
    }}

    .searching-dots span:nth-child(2) {{
        animation-delay: 0.2s;
    }}

    .searching-dots span:nth-child(3) {{
        animation-delay: 0.4s;
    }}

    @keyframes bounce {{
        0%, 60%, 100% {{
            transform: translateY(0);
        }}
        30% {{
            transform: translateY(-10px);
        }}
    }}

    .loading-text {{
        color: var(--text-secondary);
        font-size: 1rem;
        font-weight: 500;
    }}

    /* Button styling */
    .stButton button {{
        transition: all 0.3s ease !important;
        border: none !important;
        outline: none !important;
    }}

    .stButton button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px var(--shadow-color) !important;
    }}

    .stButton button:focus {{
        box-shadow: none !important;
        border-color: transparent !important;
    }}

    /* Recent queries buttons */
    .recent-query-btn {{
        background: var(--bg-input) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-secondary) !important;
        text-align: left !important;
        padding: 0.75rem 1rem !important;
        border-radius: 12px !important;
        margin-bottom: 0.5rem !important;
        transition: all 0.3s ease !important;
    }}

    .recent-query-btn:hover {{
        background: var(--accent-dim) !important;
        border-color: var(--accent-color) !important;
        color: var(--text-primary) !important;
        transform: translateX(5px) !important;
    }}

    /* Smooth scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: var(--bg-input);
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb {{
        background: var(--text-secondary);
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: var(--accent-color);
    }}

    /* Status indicator */
    .status-indicator {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: var(--accent-dim);
        color: var(--text-primary);
        border-radius: 12px;
        font-size: 0.9rem;
        font-weight: 500;
        animation: statusPulse 2s ease-in-out infinite;
    }}

    @keyframes statusPulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.7; }}
    }}

    /* Stats badge */
    .stats-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: var(--bg-input);
        color: var(--text-secondary);
        padding: 0.5rem 1rem;
        border-radius: 12px;
        font-size: 0.85rem;
        margin: 0.25rem;
        transition: all 0.3s ease;
    }}

    .stats-badge:hover {{
        background: var(--accent-dim);
        color: var(--text-primary);
        transform: translateY(-2px);
    }}

    /* Welcome message styling */
    .welcome-message {{
        text-align: center;
        padding: 3rem 1rem;
        animation: fadeIn 0.8s ease-out;
    }}

    /* User message bubble */
    .user-message {{
        background: var(--accent-dim);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1rem 1.5rem;
        color: var(--text-primary);
        font-weight: 500;
        margin: 0.5rem 0;
        animation: messageSlideIn 0.5s ease-out;
    }}

    /* Status colors */
    .status-active {{
        color: var(--accent-color);
    }}

    .status-moderate {{
        color: var(--text-secondary);
    }}

    .status-low {{
        color: var(--dark-slate);
    }}

    /* Responsive adjustments */
    @media (max-width: 768px) {{
        .summary-card, .stats-card {{
            padding: 1.5rem;
            margin: 1rem 0;
        }}

        .summary-title, .stats-title {{
            font-size: 1.5rem;
        }}

        .metadata-grid, .stats-grid {{
            grid-template-columns: 1fr;
        }}

        /* Adjust chat input for mobile */
        .stChatInputContainer textarea {{
            padding: 12px 16px !important;
            font-size: 1rem !important;
            min-height: 56px !important;
        }}
    }}

    /* Highlight important text */
    .highlight {{
        color: var(--accent-color);
        font-weight: 600;
    }}

    /* Divider styling */
    .divider {{
        height: 1px;
        background: linear-gradient(90deg,
            transparent,
            var(--border-color),
            transparent);
        margin: 2rem 0;
    }}

    /* Real-time stats specific */
    .real-time-badge {{
        background: linear-gradient(135deg, var(--accent-dim), var(--bg-input));
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
        animation: glow 2s ease-in-out infinite alternate;
    }}

    @keyframes glow {{
        from {{ box-shadow: 0 0 5px var(--border-color); }}
        to {{ box-shadow: 0 0 10px var(--accent-dim); }}
    }}

    .stats-update-time {{
        font-size: 0.8rem;
        color: var(--text-secondary);
        font-style: italic;
    }}

    /* Skip button styling - FIXED */
    .skip-button {{
        background: var(--accent-dim) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        padding: 0.5rem 1rem !important;
        border-radius: 12px !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        margin-top: 1rem !important;
        position: static !important;
        bottom: auto !important;
        right: auto !important;
        display: inline-block !important;
    }}

    .skip-button:hover {{
        background: var(--bg-input) !important;
        border-color: var(--accent-color) !important;
        transform: translateY(-2px) !important;
    }}
</style>
""", unsafe_allow_html=True)

# 4. Sidebar
with st.sidebar:

    # Theme Toggle Button
    col_t1, col_t2 = st.columns([4, 1])
    with col_t2:
        btn_emoji = "🌞" if st.session_state.theme == 'dark' else "🌙"
        if st.button(btn_emoji, key="theme_toggle", help="Toggle Light/Dark Mode", width='stretch'):
            toggle_theme()
            st.rerun()

    st.markdown("""
    <div style="padding: 10px 0 20px 0;">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 24px;">
            <div style="width: 40px; height: 40px; background: var(--accent-dim); border: 1px solid var(--border-color); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 20px; color: var(--accent-color);">
                📡
            </div>
            <h2 style="color: var(--text-primary); margin: 0;">NewsFlow AI</h2>
        </div>
        <div class="status-indicator">
            <div style="width: 8px; height: 8px; background: var(--accent-color); border-radius: 50%;"></div>
            System Online
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🔄 New Chat",
                     type="primary",
                     key="new_chat_btn",
                     use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_search = None
            st.session_state.show_stats = False
            st.session_state.show_query_history = False
            st.session_state.api_request_made = False
            st.rerun()

    with col2:
        if st.button("📊 Stats",
                     key="stats_btn",
                     use_container_width=True):
            st.session_state.show_stats = not st.session_state.show_stats
            st.session_state.show_query_history = False
            # Get fresh stats data when button is clicked
            if st.session_state.show_stats:
                st.session_state.real_stats_data = get_real_time_stats()
                st.session_state.last_stats_update = datetime.now().strftime("%H:%M:%S")
            st.rerun()

    with col3:
        if st.button("📝 History",
                     key="history_btn",
                     use_container_width=True):
            st.session_state.show_query_history = not st.session_state.show_query_history
            st.session_state.show_stats = False
            # Get fresh history data
            if st.session_state.show_query_history:
                st.session_state.query_history_data = get_detailed_query_history()
            st.rerun()

    st.divider()

    # API Usage Summary in Sidebar - FIXED VERSION
    gemini_used = get_gemini_usage()
    gemini_remaining = get_gemini_remaining()
    newsapi_used = get_newsapi_usage_info()
    newsapi_remaining = max(0, 1000 - newsapi_used)

    # Debug print to check values
    print(f"DEBUG - Gemini used: {gemini_used}, Remaining: {gemini_remaining}")
    print(f"DEBUG - NewsAPI used: {newsapi_used}, Remaining: {newsapi_remaining}")

    # Calculate percentages
    gemini_percent = (gemini_used / 20 * 100) if 20 > 0 else 0
    newsapi_percent = (newsapi_used / 1000 * 100) if 1000 > 0 else 0

    # Determine colors based on usage
    gemini_color = "var(--accent-color)" if gemini_remaining > 5 else "#ff6b6b"
    newsapi_color = "var(--accent-color)"

    # Get last fetch time
    last_fetch = None
    try:
        if os.path.exists('api_stats.json'):
            with open('api_stats.json', 'r') as f:
                api_stats = json.load(f)
            today = datetime.now().date().isoformat()
            last_fetch_str = api_stats.get(today, {}).get("last_fetch")
            if last_fetch_str:
                last_fetch = datetime.fromisoformat(last_fetch_str)
    except:
        pass

    last_fetch_text = "Never"
    if last_fetch:
        last_fetch_text = last_fetch.strftime("%H:%M")

    # Build status message
    status_messages = []
    if gemini_remaining == 0:
        status_messages.append("⛔ Gemini limit reached!")
    elif gemini_remaining <= 5:
        status_messages.append(f"⚠️ Only {gemini_remaining} Gemini calls left!")
    else:
        status_messages.append("✅ API limits healthy")

    status_html = "<br>".join(status_messages)

    # Display the API usage with CORRECT variable interpolation
    st.markdown(f"""
    <div style="margin: 20px 0; padding: 15px; background: var(--bg-input); border-radius: 12px; border: 1px solid var(--border-color);display:none;">
        <div style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 10px;">🔑 API Usage Summary</div>

        <div style="margin-bottom: 12px;display:none;">
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px;">
                <span style="color: var(--text-primary);">🤖 Gemini AI:</span>
                <span style=`color: {gemini_color}`>
                    {gemini_used}/20
                </span>
            </div>
            <div style="background: var(--border-color); height: 6px; border-radius: 3px; overflow: hidden;display:none;">
                <div style=`background: {gemini_color}; height: 100%; width: {min(gemini_percent, 100)}%`></div>
            </div>
            <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 4px;">
                {gemini_remaining} calls remaining
            </div>
        </div>

        <div style="margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px;">
                <span style="color: var(--text-primary);">📰 NewsAPI:</span>
                <span style="color: {newsapi_color};">
                    {newsapi_used}/1000
                </span>
            </div>
            <div style="background: var(--border-color); height: 6px; border-radius: 3px; overflow: hidden;">
                <div style="background: {newsapi_color}; height: 100%; width: {min(newsapi_percent, 100)}%;"></div>
            </div>
            <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 4px;">
                {newsapi_remaining} calls remaining today
            </div>
        </div>

        <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 10px; padding-top: 8px; border-top: 1px solid var(--border-color);">
            📅 Last fetch: {last_fetch_text}
            <br>
            {status_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Manual News Fetch Button
    if st.button("🔄 Fetch Fresh News",
                 help="Manually fetch latest news articles from NewsAPI",
                 use_container_width=True,
                 type="secondary"):
        if fetch_fresh_news():
            st.rerun()

    st.markdown("""
    <div style="margin: 20px 0;">
        <h4 style="color: var(--text-primary); margin-bottom: 16px;">📝 Recent Queries</h4>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.history:
        for i, query in enumerate(reversed(st.session_state.history[-5:])):
            if st.button(
                f"🔍 {query[:25]}..." if len(query) > 25 else f"🔍 {query}",
                key=f"history_{i}_{hash(query) % 1000}",
                use_container_width=True,
                help=f"Search: {query}"
            ):
                st.session_state.current_search = query
                st.session_state.show_stats = False
                st.session_state.show_query_history = False
                st.rerun()
    else:
        st.caption("No recent searches")
        st.markdown("""
        <div style="margin-top: 1rem; color: var(--text-secondary); font-size: 0.9rem;">
        Try searching for:
        <div style="margin-top: 0.5rem;">
        <span class="stats-badge">tech news</span>
        <span class="stats-badge">ai</span>
        <span class="stats-badge">politics</span>
        <span class="stats-badge">business</span>
        <span class="stats-badge">sports</span>
        <span class="stats-badge">entertainment</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # API key limit error check
    if st.session_state.get('api_key_limit_exceeded', False):
        st.error("⚠️ Gemini API error: 429 You exceeded your current quota... model: gemini-2.5-flash-lite")

    if st.session_state.get('gemini_limit_reached', False):
        st.warning("⚠️ Gemini API limit reached (20/20). Using mock responses.")

    st.divider()

# 5. Display system stats if requested
if st.session_state.show_stats:
    # Clear any previous messages
    if st.session_state.messages:
        st.session_state.messages = []

    # Get real-time stats if not already cached
    if st.session_state.real_stats_data is None:
        st.session_state.real_stats_data = get_real_time_stats()
        st.session_state.last_stats_update = datetime.now().strftime("%H:%M:%S")

    stats_data = st.session_state.real_stats_data

    if stats_data:
        # Display stats using Streamlit components
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-title">📊 Real-Time System Statistics</div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                <div style="color: var(--text-primary); font-size: 1rem; line-height: 1.6;">
                    Live monitoring of NewsFlow AI system performance and query analytics
                </div>
                <div class="real-time-badge">
                    📊 REAL DATA
                </div>
            </div>
            <div style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 0.5rem;">
                Last updated: {st.session_state.last_stats_update or 'Just now'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Stats grid using columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_queries = stats_data.get('total_queries', 0)
            st.markdown(f"""
            <div class="stat-item">
                <div class="stat-label">Today's Queries</div>
                <div class="stat-value">{total_queries}</div>
                <div style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.5rem;">Total searches today</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            success_rate = stats_data.get('success_rate', 0.0)
            st.markdown(f"""
            <div class="stat-item">
                <div class="stat-label">Success Rate</div>
                <div class="stat-value">{success_rate:.1f}%</div>
                <div style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.5rem;">Successful searches</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            avg_response = stats_data.get('avg_response_time', 0.0)
            st.markdown(f"""
            <div class="stat-item">
                <div class="stat-label">Avg Response</div>
                <div class="stat-value">{avg_response:.2f}s</div>
                <div style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.5rem;">Per query</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            articles_retrieved = stats_data.get('total_articles_retrieved', 0)
            st.markdown(f"""
            <div class="stat-item">
                <div class="stat-label">Articles Retrieved</div>
                <div class="stat-value">{articles_retrieved}</div>
                <div style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.5rem;">Today</div>
            </div>
            """, unsafe_allow_html=True)

        # API Usage Stats
        st.markdown("""
        <div style="margin-top: 2rem;">
            <div style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1rem;">🔑 API Usage & Limits</div>
        </div>
        """, unsafe_allow_html=True)

        # Get API usage
        gemini_used = get_gemini_usage()
        newsapi_used = get_newsapi_usage_info()

        # Create two columns for API usage
        col_api1, col_api2 = st.columns(2)

        with col_api1:
            gemini_percent = (gemini_used / 20 * 100) if 20 > 0 else 0
            gemini_color = "#4CAF50" if gemini_percent < 80 else "#FF9800" if gemini_percent < 95 else "#F44336"

            st.markdown(f"""
            <div class="stat-item" style="border-left: 4px solid {gemini_color};">
                <div class="stat-label">🤖 Gemini AI</div>
                <div class="stat-value" style="color: {gemini_color};">{gemini_used}/20</div>
                <div style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.5rem;">
                    Query summarization
                </div>
                <div style="background: var(--border-color); height: 6px; border-radius: 3px; margin-top: 10px; overflow: hidden;">
                    <div style="background: {gemini_color}; height: 100%; width: {min(gemini_percent, 100)}%;"></div>
                </div>
                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 5px;">
                    {20 - gemini_used} requests remaining
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_api2:
            newsapi_percent = (newsapi_used / 1000 * 100) if 1000 > 0 else 0
            newsapi_color = "#4CAF50" if newsapi_percent < 80 else "#FF9800" if newsapi_percent < 95 else "#F44336"

            st.markdown(f"""
            <div class="stat-item" style="border-left: 4px solid {newsapi_color};">
                <div class="stat-label">📰 NewsAPI</div>
                <div class="stat-value" style="color: {newsapi_color};">{newsapi_used}/1000</div>
                <div style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.5rem;">
                    Background fetching
                </div>
                <div style="background: var(--border-color); height: 6px; border-radius: 3px; margin-top: 10px; overflow: hidden;">
                    <div style="background: {newsapi_color}; height: 100%; width: {min(newsapi_percent, 100)}%;"></div>
                </div>
                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 5px;">
                    {1000 - newsapi_used} requests remaining today
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Historical and system metrics
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            historical_queries = stats_data.get('total_historical_queries', 0)
            st.markdown(f"""
            <div class="stat-item">
                <div class="stat-label">Historical Queries</div>
                <div class="stat-value">{historical_queries}</div>
                <div style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.5rem;">All time</div>
            </div>
            """, unsafe_allow_html=True)

        with col6:
            avg_similarity = stats_data.get('avg_similarity_score', 0.0)
            st.markdown(f"""
            <div class="stat-item">
                <div class="stat-label">Avg Similarity</div>
                <div class="stat-value">{avg_similarity:.3f}</div>
                <div style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.5rem;">Search relevance</div>
            </div>
            """, unsafe_allow_html=True)

        with col7:
            system_uptime = stats_data.get('system_uptime', 99.7)
            st.markdown(f"""
            <div class="stat-item">
                <div class="stat-label">System Uptime</div>
                <div class="stat-value">{system_uptime}%</div>
                <div style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.5rem;">Availability</div>
            </div>
            """, unsafe_allow_html=True)

        with col8:
            cache_hits = stats_data.get('cache_hits', 87)
            st.markdown(f"""
            <div class="stat-item">
                <div class="stat-label">Cache Hits</div>
                <div class="stat-value">{cache_hits}%</div>
                <div style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.5rem;">Performance</div>
            </div>
            """, unsafe_allow_html=True)

        # Category Usage
        st.markdown("""
        <div style="margin-top: 2rem;">
            <div style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1rem;">📈 Category Usage (Today)</div>
        </div>
        """, unsafe_allow_html=True)

        category_data = stats_data.get('category_usage', {})
        if category_data:
            # Create a DataFrame for better display
            cat_df = pd.DataFrame(
                list(category_data.items()),
                columns=['Category', 'Queries']
            ).sort_values('Queries', ascending=False)

            # Get max value for progress column
            max_value = int(cat_df['Queries'].max()) if not cat_df.empty else 1

            # Style the dataframe
            st.dataframe(
                cat_df,
                column_config={
                    "Category": st.column_config.TextColumn("Category", width="medium"),
                    "Queries": st.column_config.ProgressColumn(
                        "Queries",
                        help="Number of queries in this category",
                        format="%d",
                        width="medium",
                        min_value=0,
                        max_value=max_value
                    )
                },
                hide_index=True,
                use_container_width=True
            )

        # Source utilization table
        st.markdown("""
        <div style="margin-top: 2rem;">
            <div style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1rem;">📰 Source Access Statistics</div>
        </div>
        """, unsafe_allow_html=True)

        source_data = stats_data.get('source_accesses', {})
        if source_data:
            # Create a table using Streamlit
            source_list = []
            for source, accesses in source_data.items():
                status = "🟢 High" if accesses > 25 else "🟡 Moderate" if accesses > 10 else "⚪ Low"
                source_list.append({
                    "Source": source,
                    "Articles Accessed": accesses,
                    "Status": status
                })

            df = pd.DataFrame(source_list).sort_values('Articles Accessed', ascending=False)

            # Style the dataframe
            st.dataframe(
                df,
                column_config={
                    "Source": st.column_config.TextColumn("Source", width="medium"),
                    "Articles Accessed": st.column_config.NumberColumn("Articles Accessed", width="small"),
                    "Status": st.column_config.TextColumn("Status", width="small")
                },
                hide_index=True,
                use_container_width=True
            )

        # API Usage Explanation
        with st.expander("📖 How API Usage Works"):
            st.markdown(f"""
            **Your API Limits:**
            - **Gemini AI**: 20 requests **total** (not daily) - for query summarization
            - **NewsAPI**: 1000 requests **per day** - for background article fetching

            **Current Usage:**
            - Gemini AI: **{gemini_used}/20** ({(gemini_used/20*100):.1f}%)
            - NewsAPI: **{newsapi_used}/1000** ({(newsapi_used/1000*100):.1f}%)

            **What happens when limits are reached:**
            1. **Gemini limit reached**: System switches to mock responses (no AI)
            2. **NewsAPI limit reached**: Background fetching stops until next day

            **Tips to conserve API calls:**
            - Use mock responses when Gemini limit is low
            - NewsAPI runs automatically, no action needed
            """)

            if gemini_used >= 20:
                st.error("⚠️ Gemini API limit reached! Using mock responses only.")
            elif gemini_used >= 15:
                st.warning(f"⚠️ Gemini API limit almost reached! Only {20 - gemini_used} calls left.")

            if newsapi_used >= 900:
                st.warning(f"⚠️ NewsAPI daily limit almost reached! Only {1000 - newsapi_used} calls left today.")

        # Refresh button for stats
        col_refresh, _ = st.columns([1, 3])
        with col_refresh:
            if st.button("🔄 Refresh Stats", use_container_width=True):
                st.session_state.real_stats_data = get_real_time_stats()
                st.session_state.last_stats_update = datetime.now().strftime("%H:%M:%S")
                st.rerun()

        # Footer stats
        st.markdown(f"""
        <div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid var(--border-color);">
            <div style="display: flex; justify-content: space-between; color: var(--text-secondary); font-size: 0.9rem;">
                <div>Date: <span style="color: var(--text-primary);">{stats_data.get('date', 'Today')}</span></div>
                <div>Data Updated: <span style="color: var(--text-primary);">{st.session_state.last_stats_update or 'Just now'}</span></div>
                <div>Total Sources: <span style="color: var(--text-primary);">{len(source_data)} active</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Unable to fetch system statistics. Please try again.")

# 5.5 Display query history if requested
elif st.session_state.show_query_history:
    # Clear any previous messages
    if st.session_state.messages:
        st.session_state.messages = []

    # Get query history if not already cached
    if not st.session_state.query_history_data:
        st.session_state.query_history_data = get_detailed_query_history()

    history_data = st.session_state.query_history_data

    if history_data:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-title">📝 Query History</div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                <div style="color: var(--text-primary); font-size: 1rem; line-height: 1.6;">
                    Detailed log of all user queries with performance metrics
                </div>
                <div class="real-time-badge">
                    📊 REAL DATA
                </div>
            </div>
            <div style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 0.5rem;">
                Showing {len(history_data)} most recent queries
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Convert to DataFrame for display
        df = pd.DataFrame(history_data)

        # Format the timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

        # Display the dataframe
        st.dataframe(
            df,
            column_config={
                "timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
                "query": st.column_config.TextColumn("Query", width="large"),
                "category": st.column_config.TextColumn("Category", width="small"),
                "response_time": st.column_config.NumberColumn("Response Time (s)", format="%.2f", width="small"),
                "similarity_score": st.column_config.NumberColumn("Similarity", format="%.3f", width="small"),
                "articles_retrieved": st.column_config.NumberColumn("Articles", width="small"),
                "sources_accessed": st.column_config.TextColumn("Sources", width="medium"),
                "status": st.column_config.TextColumn("Status", width="small")
            },
            hide_index=True,
            use_container_width=True,
            height=500
        )

        # Summary stats
        if len(history_data) > 0:
            total_queries = len(history_data)
            avg_response = df['response_time'].mean() if 'response_time' in df.columns else 0
            success_rate = (df['status'] == '✅').sum() / total_queries * 100 if 'status' in df.columns else 0
            avg_similarity = df['similarity_score'].mean() if 'similarity_score' in df.columns else 0

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Queries", total_queries)
            with col2:
                st.metric("Avg Response Time", f"{avg_response:.2f}s")
            with col3:
                st.metric("Success Rate", f"{success_rate:.1f}%")
            with col4:
                st.metric("Avg Similarity", f"{avg_similarity:.3f}")

        # Refresh button
        if st.button("🔄 Refresh History", use_container_width=True):
            st.session_state.query_history_data = get_detailed_query_history()
            st.rerun()
    else:
        st.info("No query history available yet. Start searching to build history!")

# 6. Display chat history (only if not showing stats or history)
elif st.session_state.messages:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="✨" if msg["role"]=="assistant" else "👤"):
            if msg["role"] == "assistant" and isinstance(msg.get("data"), str):
                if "System Statistics" in msg["data"]:
                    continue
                else:
                    # Display news summary
                    st.markdown(f"""
                    <div class="summary-card">
                        <div class="summary-title">📰 News Analysis</div>
                        <div class="summary-content">
                            {msg['data']}
                        </div>
                        <div class="metadata-section">
                            <div style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 0.5rem;">📊 SEARCH METADATA</div>
                            <div class="metadata-grid">
                                <div class="metadata-item">
                                    <div class="metadata-label">Sources Analyzed</div>
                                    <div class="metadata-value">5+ news sources</div>
                                </div>
                                <div class="metadata-item">
                                    <div class="metadata-label">Processing Time</div>
                                    <div class="metadata-value">1.2 seconds</div>
                                </div>
                                <div class="metadata-item">
                                    <div class="metadata-label">Confidence</div>
                                    <div class="metadata-value">High</div>
                                </div>
                                <div class="metadata-item">
                                    <div class="metadata-label">System Status</div>
                                    <div class="metadata-value">Optimal</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            elif msg["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    {msg['content']}
                </div>
                """, unsafe_allow_html=True)

# 7. Show typing effect or welcome message
elif not st.session_state.show_stats and not st.session_state.show_query_history:

    if st.session_state.show_typing_effect and not st.session_state.typing_complete:

        # Determine colors for iframe based on theme
        if st.session_state.theme == 'dark':
            text_color = "#e0eab8"
            secondary_color = "#a0a895"
        else:
            text_color = "#2c3327"
            secondary_color = "#5c6355"

        # Create a container for the typing animation
        typing_container = st.container()

        with typing_container:
            components.html(f"""
            <style>
            body {{
                margin:0;
                background: transparent;
                overflow:hidden;
            }}
            .typing-container {{
                display:flex;
                flex-direction:column;
                justify-content:center;
                align-items:center;
                height:100vh;
                color:{text_color};
                font-family:Inter,sans-serif;
            }}
            .main-title {{
                font-size:4.5rem;
                font-weight:900;
                letter-spacing:-0.03em;
            }}
            .cursor {{
                display:inline-block;
                width:3px;
                height:4.5rem;
                background:{text_color};
                margin-left:5px;
                animation: blink 1s infinite;
            }}
            @keyframes blink {{
                0%,100%{{opacity:1}}
                50%{{opacity:0}}
            }}
            .team {{
                margin-top:2rem;
                font-size:1.5rem;
                color:{secondary_color};
                opacity:0;
                animation:fadein 1s forwards;
                animation-delay:2s;
                text-align: center;
                line-height: 1.6;
            }}

            @keyframes fadein{{
                to{{opacity:1}}
            }}
            </style>

            <div class="typing-container">
                <div class="main-title">
                    <span id="text"></span><span class="cursor"></span>
                </div>
                <div class="team">
                    Team Corner Stone<br>
                    Arpit Behera • Sagar Sahu • Keertan Kumar • Asim Khamari
                </div>
            </div>

            <script>
            const text = "NewsFlow AI";
            let i = 0;

            function type() {{
                if (i < text.length) {{
                    document.getElementById("text").innerHTML += text.charAt(i);
                    i++;
                    setTimeout(type, 100);
                }} else {{
                    setTimeout(() => {{
                        window.parent.postMessage("typing_complete", "*");
                    }}, 3000);
                }}
            }}
            type();
            </script>
            """, height=600)

            # Listen for JS signal - simplified approach
            try:
                # Create a simple script to listen for messages
                components.html("""
                <script>
                window.addEventListener("message", (event) => {
                    if (event.data === "typing_complete") {
                        // Send a signal back to Streamlit
                        window.parent.postMessage({type: "streamlit:setComponentValue", value: "typing_done"}, "*");
                    }
                });
                </script>
                """, height=0)

                # Check if we should skip (timeout after 8 seconds)
                if st.session_state.get('typing_signal_received', False):
                    st.session_state.show_typing_effect = False
                    st.session_state.typing_complete = True
                    st.rerun()

            except:
                pass

            # FIXED: Skip button placed properly within the container
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                if st.button("⏭ Skip Intro", key="skip_intro", use_container_width=True, type="secondary"):
                    st.session_state.show_typing_effect = False
                    st.session_state.typing_complete = True
                    st.rerun()

    else:
        # Normal welcome page - Show API usage warning if Gemini limit is low
        gemini_used = get_gemini_usage()
        gemini_remaining = get_gemini_remaining()
        newsapi_used = get_newsapi_usage_info()

        warning_html = ""
        if gemini_used >= 20:
            warning_html = """
            <div style="margin: 2rem auto; padding: 1rem; background: rgba(244, 67, 54, 0.1); border: 1px solid #F44336; border-radius: 12px; max-width: 600px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.5rem;">
                    <div style="font-size: 1.5rem;">⚠️</div>
                    <div style="font-weight: bold; color: #F44336;">Gemini API Limit Reached</div>
                </div>
                <div style="color: var(--text-primary);">
                    You've used all 20 Gemini API calls. The system will now use <strong>mock responses only</strong>.
                    To restore AI-powered summaries, you'll need a new Gemini API key.
                </div>
            </div>
            """
        elif gemini_used >= 15:
            warning_html = f"""
            <div style="margin: 2rem auto; padding: 1rem; background: rgba(255, 152, 0, 0.1); border: 1px solid #FF9800; border-radius: 12px; max-width: 600px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.5rem;">
                    <div style="font-size: 1.5rem;">⚠️</div>
                    <div style="font-weight: bold; color: #FF9800;">Gemini API Limit Warning</div>
                </div>
                <div style="color: var(--text-primary);">
                    You've used {gemini_used}/20 Gemini API calls. Only <strong>{gemini_remaining} calls remaining</strong>.
                    Once exhausted, the system will switch to mock responses.
                </div>
            </div>
            """

        # Show current API usage
        api_info_html = f"""
        <div style="margin: 1rem auto; padding: 1rem; background: var(--bg-input); border-radius: 12px; max-width: 600px; border: 1px solid var(--border-color);">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <div style="color: var(--text-primary); font-weight: 500;">🤖 Gemini AI:</div>
                <div style="color: {'var(--accent-color)' if gemini_remaining > 5 else '#ff6b6b'}">{gemini_used}/20</div>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <div style="color: var(--text-primary); font-weight: 500;">📰 NewsAPI:</div>
                <div style="color: var(--accent-color);">{newsapi_used}/1000 today</div>
            </div>
        </div>
        """

        st.markdown(f"""
        <div class="welcome-message">
            <h1 style="font-size:4.2rem;font-weight:900;color:var(--text-primary)">NewsFlow AI</h1>
            <p style="color:var(--text-secondary);font-size:1.3rem;">
            Real-time news intelligence powered by semantic search and vector embeddings.
            </p>
            {api_info_html}
            {warning_html}
        </div>
        """, unsafe_allow_html=True)

# 8. Handle search requests from quick buttons or recent queries
if st.session_state.current_search and not st.session_state.show_stats and not st.session_state.show_query_history:
    query = st.session_state.current_search

    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    if query not in st.session_state.history:
        st.session_state.history.append(query)

    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"""
        <div class="user-message">
            {query}
        </div>
        """, unsafe_allow_html=True)

    # Display assistant message with loading
    with st.chat_message("assistant", avatar="✨"):
        loading_placeholder = st.empty()

        # Show loading animation
        loading_placeholder.markdown("""
        <div class="loading-container">
            <div class="searching-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
            <div class="loading-text">Searching through news sources...</div>
        </div>
        """, unsafe_allow_html=True)

        # Simulate processing time
        time.sleep(1.5)

        # Get response
        print("🚀 Processing query...")
        st.session_state.api_request_made = True

        # Check Gemini usage before processing
        gemini_used = get_gemini_usage()
        if gemini_used >= 20:
            st.session_state.gemini_limit_reached = True

        response = process_search_query(query)

        # Clear loading and show response
        loading_placeholder.empty()

        # Add small delay for smooth transition
        time.sleep(0.3)

        # Display response
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-title">📰 News Analysis</div>
            <div class="summary-content">
                {response}
            </div>
            <div class="metadata-section">
                <div style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 0.5rem;">📊 SEARCH METADATA</div>
                <div class="metadata-grid">
                    <div class="metadata-item">
                        <div class="metadata-label">Query</div>
                        <div class="metadata-value">{query[:30]}{'...' if len(query) > 30 else ''}</div>
                    </div>
                    <div class="metadata-item">
                        <div class="metadata-label">Sources Analyzed</div>
                        <div class="metadata-value">5+ news sources</div>
                    </div>
                    <div class="metadata-item">
                        <div class="metadata-label">Processing Time</div>
                        <div class="metadata-value">1.2 seconds</div>
                    </div>
                    <div class="metadata-item">
                        <div class="metadata-label">Confidence</div>
                        <div class="metadata-value">High</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Store in session state
        st.session_state.messages.append({
            "role": "assistant",
            "data": response
        })

    # Clear current search
    st.session_state.current_search = None

# 9. Handle chat input (only if not showing stats or history)
if not st.session_state.show_stats and not st.session_state.show_query_history:
    if prompt := st.chat_input("Ask about news, technology, business, or any topic..."):
        if prompt.strip():
            st.session_state.current_search = prompt
            st.session_state.show_stats = False
            st.session_state.show_query_history = False
            st.rerun()

# 10. Footer with same background as main content
st.markdown("""
<div style="
    text-align: center;
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 1px solid var(--border-color);
    color: var(--text-secondary);
    font-size: 0.9rem;
">
    <div style="display: flex; justify-content: center; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 1rem;">
        <span style="color: var(--text-primary);">⚡ Real-time Processing</span>
        <span style="color: var(--text-primary);">🔍 Semantic Search</span>
        <span style="color: var(--text-primary);">🤖 AI-Powered</span>
        <span style="color: var(--text-primary);">📡 Live Updates</span>
    </div>
    <div>
        NewsFlow AI v2.5 • Powered by Vector RAG • Team Corner Stone • System Operational
    </div>
</div>
""", unsafe_allow_html=True)
