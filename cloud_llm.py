"""
llm_router.py
Routes between local Ollama and cloud Gemini LLMs
"""

import os

# Read provider from environment
PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

print(f"🔀 LLM provider selected: {PROVIDER}")

# ======================================================
# Load the correct LLM
# ======================================================

if PROVIDER == "gemini":
    from cloud_llm import gemini_llm
    llm = gemini_llm
    print("☁️ Using Gemini Cloud LLM")

elif PROVIDER == "ollama":
    from ollama_llm import local_llm
    llm = local_llm
    print("🧠 Using Ollama Local LLM")

else:
    raise RuntimeError(f"❌ Unknown LLM_PROVIDER: {PROVIDER}")
