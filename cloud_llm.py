import os
import google.generativeai as genai  # FIXED IMPORT

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not set")

genai.configure(api_key=API_KEY)

MODEL = "gemini-1.5-flash"  # Updated to a valid model name

def llm(prompt: str, history=None):
    model = genai.GenerativeModel(MODEL)
    response = model.generate_content(prompt)
    return response.text
