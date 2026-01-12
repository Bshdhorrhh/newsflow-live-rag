import os
import google.generativeai as genai  # CORRECT IMPORT

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not set")

genai.configure(api_key=API_KEY)

# Use correct model name for gemini-2.5-flash-lite
MODEL = "gemini-2.5-flash-lite"

def llm(prompt: str, history=None):
    """Simple LLM call with Gemini"""
    try:
        model = genai.GenerativeModel(MODEL)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"⚠️ Gemini API error in cloud_llm: {e}")
        return f"Error: Unable to generate response. {str(e)}"
