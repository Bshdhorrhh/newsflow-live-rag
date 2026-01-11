import os
from google import genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not set")

# Create Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

MODEL = "gemini-2.5-flash-lite"

def ask_gemini(prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        return response.text.strip()

    except Exception as e:
        return f"Gemini Error: {str(e)}"
print("☁️ Gemini client ready:", MODEL)
