import os
import time

# Try to get API key from environment
API_KEY = os.getenv("GEMINI_API_KEY")

# Mock response for when API key is not available
MOCK_RESPONSES = [
    "Based on the latest news analysis, there are significant developments in this area with multiple sources reporting.",
    "Recent coverage indicates growing interest and activity around this topic across various news outlets.",
    "Analysis of current articles shows evolving trends and new information emerging daily.",
    "Multiple perspectives are available on this subject, with experts offering diverse insights.",
    "Breaking news and ongoing developments make this a dynamic area of coverage."
]

def llm_answer(prompt: str) -> str:
    """Generate answer using Gemini or fallback to mock responses"""

    if not API_KEY:
        # Return mock response if no API key
        import random
        time.sleep(0.5)  # Simulate processing time
        return random.choice(MOCK_RESPONSES)

    try:
        # Configure Gemini
        import google.generativeai as genai
        genai.configure(api_key=API_KEY)

        # Use the model
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)

        return response.text

    except Exception as e:
        print(f"⚠️ Gemini API error: {e}")
        # Fallback to mock response
        import random
        return random.choice(MOCK_RESPONSES)
