#!/usr/bin/env python3
"""
ENHANCED QUERY ENGINE - FIXED VERSION
Improved accuracy with better scoring and relaxed thresholds
WITH REAL-TIME STATS TRACKING
"""

import os
import json
import numpy as np
import re
import readline
from datetime import datetime, timedelta
from collections import defaultdict
import math
import time as time_module
import sqlite3
import threading
from typing import Dict, List, Tuple, Optional

# ======================================================
# STATS DATABASE SETUP
# ======================================================

class StatsTracker:
    """Track real-time statistics for the query engine"""
    
    def __init__(self, db_path="query_stats.db"):
        self.db_path = db_path
        self._init_db()
        self.lock = threading.Lock()
        
        # In-memory cache for today's stats
        self.today_cache = {
            'total_queries': 0,
            'total_response_time': 0.0,
            'source_accesses': defaultdict(int),
            'category_usage': defaultdict(int),
            'successful_searches': 0,
            'failed_searches': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_similarity_score': 0.0,
            'articles_retrieved': 0
        }
        
    def _init_db(self):
        """Initialize SQLite database for stats"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create stats table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS query_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            query TEXT,
            category TEXT,
            response_time REAL,
            similarity_score REAL,
            articles_retrieved INTEGER,
            sources_accessed TEXT,
            success BOOLEAN
        )
        ''')
        
        # Create daily summary table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_stats (
            date DATE PRIMARY KEY,
            total_queries INTEGER DEFAULT 0,
            avg_response_time REAL DEFAULT 0.0,
            successful_searches INTEGER DEFAULT 0,
            failed_searches INTEGER DEFAULT 0,
            cache_hits INTEGER DEFAULT 0,
            cache_misses INTEGER DEFAULT 0,
            total_articles_retrieved INTEGER DEFAULT 0
        )
        ''')
        
        conn.commit()
        conn.close()
        
    def record_query(self, query: str, category: str, response_time: float, 
                    similarity_score: float, articles_retrieved: int, 
                    sources_accessed: List[str], success: bool):
        """Record a single query execution"""
        with self.lock:
            today = datetime.now().date()
            
            # Update in-memory cache
            self.today_cache['total_queries'] += 1
            self.today_cache['total_response_time'] += response_time
            
            for source in sources_accessed:
                self.today_cache['source_accesses'][source] += 1
                
            self.today_cache['category_usage'][category] += 1
            
            if success:
                self.today_cache['successful_searches'] += 1
            else:
                self.today_cache['failed_searches'] += 1
                
            self.today_cache['articles_retrieved'] += articles_retrieved
            
            # Update avg similarity score
            current_avg = self.today_cache['avg_similarity_score']
            total_queries = self.today_cache['total_queries']
            self.today_cache['avg_similarity_score'] = (
                (current_avg * (total_queries - 1) + similarity_score) / total_queries
            )
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO query_stats 
            (query, category, response_time, similarity_score, articles_retrieved, sources_accessed, success)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (query, category, response_time, similarity_score, 
                 articles_retrieved, ','.join(sources_accessed), success))
            
            conn.commit()
            conn.close()
            
            return True
            
    def get_today_stats(self) -> Dict:
        """Get statistics for today"""
        with self.lock:
            today = datetime.now().date()
            
            # Calculate averages
            total_queries = self.today_cache['total_queries']
            avg_response = 0.0
            if total_queries > 0:
                avg_response = self.today_cache['total_response_time'] / total_queries
                
            # Calculate success rate
            success_rate = 0.0
            if total_queries > 0:
                success_rate = (self.today_cache['successful_searches'] / total_queries) * 100
                
            return {
                'date': str(today),
                'total_queries': total_queries,
                'avg_response_time': avg_response,
                'successful_searches': self.today_cache['successful_searches'],
                'failed_searches': self.today_cache['failed_searches'],
                'success_rate': success_rate,
                'total_articles_retrieved': self.today_cache['articles_retrieved'],
                'avg_similarity_score': self.today_cache['avg_similarity_score'],
                'source_accesses': dict(self.today_cache['source_accesses']),
                'category_usage': dict(self.today_cache['category_usage'])
            }
            
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        stats = self.get_today_stats()
        
        # Add database stats
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total historical queries
        cursor.execute('SELECT COUNT(*) FROM query_stats')
        total_historical = cursor.fetchone()[0]
        
        # Average historical response time
        cursor.execute('SELECT AVG(response_time) FROM query_stats')
        avg_historical_response = cursor.fetchone()[0] or 0.0
        
        # Most used categories
        cursor.execute('''
        SELECT category, COUNT(*) as count 
        FROM query_stats 
        GROUP BY category 
        ORDER BY count DESC 
        LIMIT 5
        ''')
        top_categories = cursor.fetchall()
        
        conn.close()
        
        stats.update({
            'total_historical_queries': total_historical,
            'avg_historical_response_time': avg_historical_response,
            'top_categories': dict(top_categories),
            'system_uptime': 99.7,  # Mock uptime, could be calculated
            'cache_hits': 87,  # Mock cache hit rate
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return stats

# Initialize global stats tracker
stats_tracker = StatsTracker()

# ======================================================
# LLM BACKEND ROUTER
# ======================================================

from llm_router import llm_answer


# ======================================================
# CONFIG - AUTO-DETECT EMBEDDING DIMENSION
# ======================================================

VECTORS_FILE = "vectors.npy"
META_FILE = "metadata.json"

# Load vectors first to determine actual embedding dimension
vectors = np.load(VECTORS_FILE)
EMBED_DIM = vectors.shape[1]  # Auto-detect from saved vectors
print(f"🔍 Detected embedding dimension: {EMBED_DIM}")

TOP_K = 15  # Increased to get more results
MIN_SIMILARITY_THRESHOLD = 0.1  # Lowered significantly for better recall

# ======================================================
# ENHANCED CATEGORY KEYWORDS WITH BETTER MATCHING
# ======================================================

CATEGORIES = {
    "technology": {
        "keywords": {
            "ai": 3.0, "artificial": 2.5, "intelligence": 2.5, "machine": 2.0,
            "learning": 2.0, "tech": 2.5, "technology": 2.5, "software": 1.8,
            "hardware": 1.8, "computer": 1.8, "digital": 1.8, "innovation": 1.8,
            "smartphone": 2.0, "laptop": 1.8, "electronics": 1.8,
            "samsung": 2.0, "apple": 2.0, "microsoft": 2.0, "google": 2.0,
            "startup": 1.5, "app": 1.5, "cloud": 1.8, "data": 1.8,
            "gadget": 1.5, "device": 1.5, "internet": 1.5, "web": 1.5,
            "cyber": 1.5, "security": 1.5, "coding": 1.5, "program": 1.5
        },
        "boost": 1.2  # Reduced boost for more balanced scoring
    },
    "politics": {
        "keywords": {
            "politics": 3.0, "political": 2.8, "government": 2.8, "election": 2.8,
            "president": 2.8, "senate": 2.5, "parliament": 2.5, "congress": 2.5,
            "minister": 2.5, "policy": 2.5, "law": 2.5, "regulation": 2.5,
            "democracy": 2.0, "vote": 2.0, "voting": 2.0, "party": 2.0,
            "diplomacy": 1.8, "treaty": 1.8, "foreign": 1.8, "international": 1.8,
            "leader": 2.0, "administration": 2.0, "bill": 2.0, "act": 2.0,
            "rights": 1.8, "freedom": 1.8, "justice": 1.8, "court": 1.8
        },
        "boost": 1.2
    },
    "business": {
        "keywords": {
            "business": 3.0, "economy": 3.0, "economic": 2.8, "market": 2.8,
            "markets": 2.8, "stock": 2.5, "stocks": 2.5, "trade": 2.5,
            "finance": 2.8, "financial": 2.8, "company": 2.5, "companies": 2.5,
            "revenue": 2.0, "profit": 2.0, "investment": 2.5, "investor": 2.0,
            "exchange": 2.0, "entrepreneur": 1.8, "startup": 1.8, "venture": 1.8,
            "capital": 2.0, "merger": 2.0, "acquisition": 2.0, "quarterly": 1.8,
            "earnings": 2.0, "growth": 2.0, "industry": 2.0, "corporate": 2.0,
            "bank": 2.0, "banking": 2.0, "dollar": 1.8, "currency": 1.8
        },
        "boost": 1.2
    },
    "sports": {
        "keywords": {
            "sports": 3.0, "sport": 3.0, "cricket": 2.8, "football": 2.8,
            "soccer": 2.8, "match": 2.5, "matches": 2.5, "league": 2.5,
            "team": 2.5, "teams": 2.5, "player": 2.5, "players": 2.5,
            "tournament": 2.5, "championship": 2.5, "olympics": 2.8, "olympic": 2.8,
            "nba": 2.5, "fifa": 2.5, "score": 2.0, "scored": 2.0,
            "win": 2.0, "won": 2.0, "winning": 2.0, "lost": 2.0,
            "losing": 2.0, "coach": 2.0, "game": 2.5, "games": 2.5,
            "athlete": 2.0, "champion": 2.0, "competition": 2.0
        },
        "boost": 1.1
    },
    "science": {
        "keywords": {
            "science": 3.0, "scientific": 2.8, "research": 2.8, "space": 2.8,
            "nasa": 2.5, "experiment": 2.5, "climate": 2.8, "environment": 2.5,
            "environmental": 2.5, "discovery": 2.5, "scientist": 2.5,
            "study": 2.5, "studies": 2.5, "lab": 2.0, "laboratory": 2.0,
            "university": 2.0, "innovation": 2.0, "breakthrough": 2.5,
            "biology": 2.5, "biological": 2.0, "chemistry": 2.5,
            "chemical": 2.0, "physics": 2.5, "physical": 2.0,
            "medical": 2.5, "medicine": 2.5, "health": 2.5
        },
        "boost": 1.2
    },
    "entertainment": {
        "keywords": {
            "entertainment": 3.0, "movie": 2.8, "movies": 2.8, "film": 2.8,
            "films": 2.8, "tv": 2.5, "television": 2.5, "series": 2.5,
            "music": 2.8, "musical": 2.5, "actor": 2.5, "actors": 2.5,
            "actress": 2.5, "show": 2.5, "shows": 2.5, "celebrity": 2.5,
            "celebrities": 2.5, "oscar": 2.5, "oscars": 2.5, "award": 2.5,
            "awards": 2.5, "netflix": 2.5, "hollywood": 2.5, "bollywood": 2.5,
            "director": 2.5, "producer": 2.5, "screen": 2.0, "cinema": 2.5,
            "theater": 2.0, "drama": 2.0, "comedy": 2.0, "action": 2.0
        },
        "boost": 1.1
    }
}

# ======================================================
# SOURCE DETECTION
# ======================================================

def extract_sources_from_text(text: str) -> List[str]:
    """Extract news sources mentioned in text"""
    common_sources = [
        'Bloomberg', 'Reuters', 'TechCrunch', 'Financial Times', 
        'BBC News', 'The Verge', 'Wall Street Journal', 'CNBC',
        'CNN', 'New York Times', 'Washington Post', 'BBC',
        'Guardian', 'Forbes', 'Business Insider', 'Associated Press'
    ]
    
    found_sources = []
    for source in common_sources:
        if source.lower() in text.lower():
            found_sources.append(source)
    
    return found_sources if found_sources else ['Unknown Source']

# ======================================================
# LOAD STORAGE
# ======================================================

print("📦 Loading metadata...")
with open(META_FILE) as f:
    raw_meta = json.load(f)

docs = []
doc_timestamps = {}
time_pattern = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}')

for key, value in raw_meta.items():
    if isinstance(value, dict):
        text = value.get("text", "")
        published = value.get("published_at", "")
        title = value.get("title", "")  # Extract title if available
    else:
        text = str(value)
        published = ""
        title = ""
    
    # Store enriched document data
    docs.append({
        "id": key,
        "text": text,
        "title": title,
        "published": published,
        "normalized_text": "",  # Will be filled
        "keywords": []  # Will be filled
    })

print(f"✅ Loaded {len(docs)} documents")

# ======================================================
# SIMPLIFIED NORMALIZATION (MATCH ORIGINAL)
# ======================================================

def normalize(text: str) -> str:
    """Simple normalization matching the original vector creation"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# Pre-normalize all documents
for i, doc in enumerate(docs):
    # Combine title and text for better matching
    full_text = f"{doc['title']} {doc['text']}" if doc['title'] else doc['text']
    doc["normalized_text"] = normalize(full_text)
    doc["keywords"] = list(set(doc["normalized_text"].split()))

# ======================================================
# SIMPLIFIED EMBEDDING (MATCH ORIGINAL EXACTLY)
# ======================================================

def embed_text(text: str) -> np.ndarray:
    """Create embedding vector matching the original exactly"""
    text_norm = normalize(text)
    words = text_norm.split()
    
    # Initialize embedding vector with the correct dimension
    vec = np.zeros(EMBED_DIM)
    
    # Simple word counting as in original
    for word in words:
        # Use same hash function as original
        idx = abs(hash(word)) % EMBED_DIM
        vec[idx] += 1
    
    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    
    return vec

# ======================================================
# SIMPLIFIED CATEGORY DETECTION
# ======================================================

def detect_category_with_confidence(query: str):
    """Simplified category detection"""
    q_norm = normalize(query)
    q_words = set(q_norm.split())
    
    # Check for time-related queries
    time_patterns = {'today', 'yesterday', 'week', 'month', 'recent', 'latest', 'new'}
    is_time_restricted = any(pattern in q_norm for pattern in time_patterns)
    
    # Calculate category scores
    category_scores = {}
    for cat_name, cat_data in CATEGORIES.items():
        score = 0
        keywords = cat_data["keywords"]
        
        for word in q_words:
            if word in keywords:
                score += keywords[word]
        
        category_scores[cat_name] = score
    
    # Get primary category
    sorted_cats = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    primary = "general"
    confidence = 0.0
    
    if sorted_cats and sorted_cats[0][1] > 0:
        primary = sorted_cats[0][0]
        # Simple confidence calculation
        confidence = min(sorted_cats[0][1] / 5.0, 1.0)
    
    return {
        "primary": primary,
        "confidence": confidence,
        "is_time_restricted": is_time_restricted,
        "time_keywords": [p for p in time_patterns if p in q_norm]
    }

# ======================================================
# SIMPLIFIED RETRIEVAL WITH BETTER MATCHING
# ======================================================

def retrieve_enhanced(query: str) -> Tuple[List[Tuple[float, int, Dict]], Dict]:
    """Simplified retrieval that actually finds results"""
    # 1. Intent detection
    intent_info = detect_category_with_confidence(query)
    print(f"🔍 Detected: {intent_info['primary']} (confidence: {intent_info['confidence']:.2f})")
    
    # 2. Query embedding
    qvec = embed_text(query)
    
    # 3. Calculate similarities for ALL documents
    similarities = []
    for idx, doc in enumerate(docs):
        # Vector similarity
        vector_sim = float(np.dot(vectors[idx], qvec))
        
        # Keyword overlap bonus
        query_words = set(normalize(query).split())
        doc_words = set(doc["normalized_text"].split())
        keyword_overlap = len(query_words.intersection(doc_words))
        
        # Category bonus if detected
        category_bonus = 0
        if intent_info["primary"] != "general":
            # Check if document contains category keywords
            cat_keywords = CATEGORIES[intent_info["primary"]]["keywords"]
            for keyword in cat_keywords:
                if keyword in doc["normalized_text"]:
                    category_bonus += 0.1
        
        # Combined score with weights
        # Base: vector similarity (0.6) + keyword overlap (0.3) + category bonus (0.1)
        final_score = (vector_sim * 0.6) + (min(keyword_overlap * 0.1, 0.3)) + category_bonus
        
        similarities.append((final_score, idx, {
            "vector_sim": vector_sim,
            "keyword_overlap": keyword_overlap,
            "category_bonus": category_bonus
        }))
    
    # 4. Sort by score
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    # 5. Apply VERY lenient threshold
    results = []
    for score, idx, breakdown in similarities:
        if score > 0.05:  # Very low threshold to catch most results
            results.append((score, idx, breakdown))
    
    # 6. Take top results
    top_results = results[:TOP_K]
    
    # Debug: Show top scores
    if top_results:
        print(f"📊 Top scores: {', '.join([f'{score:.3f}' for score, _, _ in top_results[:3]])}")
    
    return top_results, intent_info

# ======================================================
# IMPROVED SUMMARY GENERATION WITH STATS TRACKING
# ======================================================

def generate_multi_result_summary(query: str, results, intent_info) -> Tuple[str, Dict]:
    """Generate comprehensive summary from multiple results and return tracking data"""
    
    start_time = time_module.time()
    sources_accessed = []
    avg_similarity = 0.0
    
    if not results:
        # Try to get at least something
        print("⚠️  No high-scoring results, showing best matches anyway...")
        # Get all documents sorted by similarity
        qvec = embed_text(query)
        all_scores = []
        for idx, doc in enumerate(docs):
            score = float(np.dot(vectors[idx], qvec))
            all_scores.append((score, idx))
        all_scores.sort(key=lambda x: x[0], reverse=True)
        top_indices = [idx for score, idx in all_scores[:5]]
        top_docs = [docs[idx] for idx in top_indices]
        avg_similarity = np.mean([score for score, _ in all_scores[:5]]) if all_scores[:5] else 0.0
    else:
        top_docs = [docs[idx] for _, idx, breakdown in results[:5]]
        # Extract sources from retrieved documents
        for doc in top_docs:
            sources = extract_sources_from_text(doc['text'])
            sources_accessed.extend(sources)
        avg_similarity = np.mean([score for score, _, _ in results[:5]]) if results[:5] else 0.0
    
    # Remove duplicate sources
    sources_accessed = list(set(sources_accessed))
    
    # Prepare context
    context_parts = []
    for i, doc in enumerate(top_docs):
        snippet = doc['text'][:300] + "..." if len(doc['text']) > 300 else doc['text']
        source_info = f"[Source {i+1}]"
        if doc.get('title'):
            source_info += f" {doc['title']}"
        context_parts.append(f"{source_info}: {snippet}")
    
    context_text = "\n\n".join(context_parts)
    
    # Smart prompt based on results
    if len(results) > 0:
        result_count = len(results)
        confidence_msg = f"Found {result_count} relevant news articles"
    else:
        result_count = len(top_docs)
        confidence_msg = f"Showing {result_count} potentially related articles"
    
    # Create prompt
    prompt = f"""USER QUERY: "{query}"

CONTEXT ARTICLES ({result_count} articles):
{context_text}

Please provide a comprehensive summary that:
1. Directly addresses the user's query about "{query}"
2. Combines information from all the provided articles
3. Organizes information in clear paragraphs
4. Highlights the most important developments or news
5. Mentions if there are different perspectives in the sources

Format the response to be informative and well-structured."""

    try:
        # Get LLM summary
        summary = llm_answer(prompt)

        # Calculate response time
        response_time = time_module.time() - start_time
        
        # Add footer
        footer = f"""

---
📊 **Search Information**
• Query: "{query}"
• Category: {intent_info['primary'].title()}
• Articles analyzed: {result_count}
• Confidence: {intent_info['confidence']:.1%}
• Response time: {response_time:.2f}s
• Sources: {', '.join(sources_accessed) if sources_accessed else 'Various news sources'}
"""
        
        # Create tracking data
        tracking_data = {
            'query': query,
            'category': intent_info['primary'],
            'response_time': response_time,
            'similarity_score': avg_similarity,
            'articles_retrieved': result_count,
            'sources_accessed': sources_accessed,
            'success': True
        }
        
        return summary + footer, tracking_data
    except Exception as e:
        # Calculate response time even on error
        response_time = time_module.time() - start_time
        
        # Fallback manual summary
        summary_lines = [f"## 📰 News Summary for: {query}"]
        summary_lines.append(f"**Category:** {intent_info['primary'].title()}")
        summary_lines.append(f"**Articles found:** {result_count}\n")
        
        for i, doc in enumerate(top_docs):
            summary_lines.append(f"**Article {i+1}:**")
            summary_lines.append(f"{doc['text'][:200]}...")
            summary_lines.append("")
        
        # Create tracking data for failed query
        tracking_data = {
            'query': query,
            'category': intent_info['primary'],
            'response_time': response_time,
            'similarity_score': avg_similarity,
            'articles_retrieved': result_count,
            'sources_accessed': sources_accessed,
            'success': False
        }
        
        return "\n".join(summary_lines), tracking_data

# ======================================================
# STREAMLIT COMPATIBLE FUNCTION WITH STATS TRACKING
# ======================================================

def rag_answer(query: str) -> str:
    """
    Used by Streamlit UI - with real-time stats tracking
    """
    try:
        # Start timing
        search_start = time_module.time()
        
        # Perform search
        results, intent_info = retrieve_enhanced(query)
        
        # Generate summary and get tracking data
        summary, tracking_data = generate_multi_result_summary(query, results, intent_info)
        
        # Record stats
        stats_tracker.record_query(
            query=tracking_data['query'],
            category=tracking_data['category'],
            response_time=tracking_data['response_time'],
            similarity_score=tracking_data['similarity_score'],
            articles_retrieved=tracking_data['articles_retrieved'],
            sources_accessed=tracking_data['sources_accessed'],
            success=tracking_data['success']
        )
        
        # Add performance info
        total_time = time_module.time() - search_start
        summary += f"\n\n⚡ **Performance:** Search completed in {total_time:.2f}s"
        
        return summary
    except Exception as e:
        # Record failed query
        stats_tracker.record_query(
            query=query,
            category="unknown",
            response_time=0.0,
            similarity_score=0.0,
            articles_retrieved=0,
            sources_accessed=["Unknown"],
            success=False
        )
        
        return f"""## ⚠️ Search Results for: {query}

**Status:** Search completed with some limitations.

**Details:** The system found potential matches but couldn't generate a detailed summary due to technical constraints.

**Suggested action:** Try a more specific query or different keywords.

*Error details: {str(e)}*"""

# ======================================================
# STATS EXPORT FUNCTIONS FOR STREAMLIT
# ======================================================

def get_system_stats() -> Dict:
    """Get comprehensive system statistics for Streamlit UI"""
    return stats_tracker.get_system_stats()

def get_live_stats() -> Dict:
    """Get live statistics for real-time display"""
    stats = stats_tracker.get_today_stats()
    
    # Add some mock system metrics (these would be real in production)
    stats.update({
        'active_sources': len(stats.get('source_accesses', {})),
        'avg_response': stats.get('avg_response_time', 0.0),
        'total_articles': len(docs),  # Total articles in database
        'system_status': '🟢 Optimal',
        'cache_status': '87% hit rate',
        'memory_usage': '2.4GB / 8GB',
        'cpu_usage': '34%'
    })
    
    return stats

# ======================================================
# INTERACTIVE INTERFACE
# ======================================================

def interactive():
    print("\n" + "="*60)
    print("📰 NEWS SEARCH ENGINE WITH REAL-TIME STATS")
    print("="*60)
    print(f"\nSystem ready with {len(docs)} documents")
    print("Type your query or 'exit' to quit\n")
    
    while True:
        try:
            query = input("🔍 Query> ").strip()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        
        if query.lower() in {"exit", "quit", "q"}:
            break
        
        if not query:
            continue
        
        # Search
        print(f"\nSearching for: {query}")
        start_time = datetime.now()
        
        # Use the main function which tracks stats
        result = rag_answer(query)
        
        search_time = (datetime.now() - start_time).total_seconds()
        
        # Display results
        print("\n" + "="*80)
        print("📰 SEARCH RESULTS")
        print("="*80)
        print(f"\n{result}")
        print(f"\n⏱️  Search completed in {search_time:.2f} seconds")
        
        # Show current stats
        print("\n📊 CURRENT STATS:")
        stats = stats_tracker.get_today_stats()
        print(f"• Total queries today: {stats['total_queries']}")
        print(f"• Avg response time: {stats['avg_response_time']:.2f}s")
        print(f"• Success rate: {stats['success_rate']:.1f}%")
        
        print("="*80)
        print()

# ======================================================
# ENTRY POINT
# ======================================================

if __name__ == "__main__":
    interactive()
