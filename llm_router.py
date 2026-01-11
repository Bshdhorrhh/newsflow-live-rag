import os

LLM_BACKEND = os.getenv("LLM_BACKEND", "local").lower()

if LLM_BACKEND == "cloud":
    from llm_cloud import ask_llm
else:
    from llm_local import ask_llm
