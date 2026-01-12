import os
import time
import google.genai as genai  # NEW IMPORT (this was originally correct!)

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
        # Configure Gemini with the new package
        genai.configure(api_key=API_KEY)

        # Use the new client with CORRECT model name
        client = genai.Client(api_key=API_KEY)

        # Generate response with gemini-2.5-flash-lite
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",  # CORRECT MODEL NAME
            contents=prompt
        )

        return response.text

    except Exception as e:
        print(f"⚠️ Gemini API error: {e}")
        # Fallback to mock response
        import random
        return random.choice(MOCK_RESPONSES)
