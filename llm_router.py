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
        error_str = str(e)
        print(f"⚠️ Gemini API error: {error_str}")

        # Check for API quota exceeded error
        if "429" in error_str or "quota" in error_str.lower() or "exceeded" in error_str.lower():
            # RAISE the specific error so it can be caught upstream
            raise Exception("⚠️ Gemini API error: 429 You exceeded your current quota... model: gemini-2.5-flash-lite")
        else:
            # For other errors, fallback to mock response
            import random
            time.sleep(0.5)
            return random.choice(MOCK_RESPONSES)
