"""
llm_router.py
Single source of truth for LLM calls
"""

import os

PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
print(f"🔀 LLM provider: {PROVIDER}")

if PROVIDER == "gemini":
    from cloud_llm import gemini_llm
    llm = gemini_llm
    print("☁️ Using Gemini")

elif PROVIDER == "ollama":
    from ollama_llm import ask_llm as llm
    print("🧠 Using Ollama")

else:
    raise RuntimeError(f"Unknown LLM_PROVIDER: {PROVIDER}")

