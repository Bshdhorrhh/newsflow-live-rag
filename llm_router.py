import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-2.5-flash-lite")

def llm_answer(prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text
