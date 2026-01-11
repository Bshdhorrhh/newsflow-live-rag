import os

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

if LLM_PROVIDER == "gemini":
    from cloud_llm import ask_gemini

    def llm(prompt, history):
        return ask_gemini(prompt)

elif LLM_PROVIDER == "ollama":
    from local_llm import ask_ollama

    def llm(prompt, history):
        return ask_ollama(prompt)

else:
    raise RuntimeError("Invalid LLM_PROVIDER")
