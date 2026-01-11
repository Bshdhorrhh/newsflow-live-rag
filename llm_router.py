from cloud_llm import gemini_generate

def llm_answer(prompt: str) -> str:
    return gemini_generate(prompt)
