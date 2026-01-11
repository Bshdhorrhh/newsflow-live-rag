"""
ollama_llm.py
Local LLM backend using Ollama (Mistral)

- Used when LLM_BACKEND=local
- Raises error if mistakenly used in cloud mode
- No OpenAI / no cloud dependency
"""

import os
import requests

# ======================================================
# ENV VALIDATION
# ======================================================

LLM_BACKEND = os.getenv("LLM_BACKEND", "local").lower()

if LLM_BACKEND == "cloud":
    raise RuntimeError(
        "❌ LLM_BACKEND is set to 'cloud' but ollama_llm.py was loaded.\n"
        "➡ Either:\n"
        "   • set LLM_BACKEND=local\n"
        "   • or configure cloud LLM API keys properly\n"
    )

# ======================================================
# OLLAMA CONFIG
# ======================================================

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# Performance tuning (CPU heavy)
NUM_THREADS = int(os.getenv("OLLAMA_THREADS", "8"))
CTX_SIZE = int(os.getenv("OLLAMA_CTX", "4096"))
MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("OLLAMA_TEMP", "0.3"))

# ======================================================
# LOCAL LLM CALL
# ======================================================

def ask_llm(query: str, context_docs: list[str]) -> str:
    """
    Query Ollama with RAG context
    """

    if not context_docs:
        return "⚠️ No context documents provided to LLM."

    context = "\n\n".join(context_docs)

    prompt = f"""
You are a factual news assistant.

Context (latest relevant news):
{context}

User question:
{query}

Answer clearly, concisely, and only using the context above.
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": CTX_SIZE,
                    "num_predict": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                    "num_thread": NUM_THREADS
                }
            },
            timeout=180
        )
    except Exception as e:
        return f"❌ Ollama connection error: {e}"

    if response.status_code != 200:
        return f"❌ Ollama error [{response.status_code}]: {response.text}"

    return response.json().get("response", "").strip() or "⚠️ Empty response from Ollama"

