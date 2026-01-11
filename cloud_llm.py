import os
from google import genai

# Read API key
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not set")

# Create Gemini client
client = genai.Client(api_key=API_KEY)

MODEL = "gemini-2.5-flash-lite"

def llm(prompt: str, history=None):
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )
        return response.text
    except Exception as e:
        print("❌ Gemini API error:", e)
        raise
