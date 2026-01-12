import os
import google.generativeai as genai
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
        genai.configure(api_key=API_KEY)

        # Use a lightweight model
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Generate response with timeout
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 500,
            }
        )

        return response.text

    except Exception as e:
        print(f"⚠️ Gemini API error: {e}")
        # Fallback to mock response
        import random
        return random.choice(MOCK_RESPONSES)
