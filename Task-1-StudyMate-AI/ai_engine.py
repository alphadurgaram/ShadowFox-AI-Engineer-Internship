import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not configured.")

# Create Gemini client
client = genai.Client(api_key=api_key)


def generate_response(prompt):
    try:
        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=prompt
        )

        return interaction.output_text

    except Exception as e:
        return f"API Error: {str(e)}"