"""
cloud_llm.py
Google Gemini Cloud LLM (Text Generation)
Used when LLM_PROVIDER=gemini
"""

import os
import google.generativeai as genai

# ======================================================
# CONFIG
# ======================================================

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not set")

# Supported fast model
MODEL_NAME = "gemini-2.5-flash-lite"

# ======================================================
# INIT
# ======================================================

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel(
    MODEL_NAME,
    generation_config={
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 512,
    },
)

print(f"☁️ Cloud LLM ready: {MODEL_NAME}")

# ======================================================
# INTERNAL GEMINI CALL
# ======================================================

def _ask_gemini(query: str, context_docs: list[str]) -> str:
    context = "\n\n".join(
        f"[DOC {i+1}]\n{doc[:1500]}"
        for i, doc in enumerate(context_docs)
    )

    prompt = f"""
You are a professional news analyst.

RULES:
- Use ONLY the context below
- Do NOT hallucinate
- If context is insufficient, say so

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini error: {e}"


# ======================================================
# PUBLIC LLM INTERFACE (for llm_router.py)
# ======================================================

def gemini_llm(query: str, context_docs: list[str]) -> str:
    return _ask_gemini(query, context_docs)
