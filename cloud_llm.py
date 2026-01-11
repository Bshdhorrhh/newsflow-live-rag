import os
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not set")

# Configure client
genai.configure(api_key=GEMINI_API_KEY)

# Gemini 2.5 Flash Lite
MODEL_NAME = "gemini-2.5-flash-lite"

model = genai.GenerativeModel(MODEL_NAME)

def ask_gemini(prompt: str) -> str:
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "top_p": 0.9,
                "max_output_tokens": 1024,
            }
        )

        return response.text.strip()

    except Exception as e:
        return f"Gemini Error: {str(e)}"
