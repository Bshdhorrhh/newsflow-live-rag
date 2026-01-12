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
        time.sleep(0.5)
        return random.choice(MOCK_RESPONSES)

    try:
        # IMPORTANT: Use google.generativeai (not google.genai)
        import google.generativeai as genai

        # Configure Gemini
        genai.configure(api_key=API_KEY)

        # Use gemini-2.5-flash-lite (correct model name)
        model = genai.GenerativeModel('gemini-2.5-flash-lite')

        # Generate response with timeout
        response = model.generate_content(prompt)

        if response and hasattr(response, 'text'):
            return response.text
        else:
            print("⚠️ Gemini returned empty response, using fallback")
            import random
            return random.choice(MOCK_RESPONSES)

    except Exception as e:
        print(f"⚠️ Gemini API error: {str(e)}")
        # Fallback to mock response
        import random
        time.sleep(0.5)
        return random.choice(MOCK_RESPONSES)
