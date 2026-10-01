import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not configured.")


client = genai.Client(api_key=api_key)


def generate_answer(question, relevant_chunks):

    context_parts = []

    for chunk in relevant_chunks:

        page_number = chunk["page"]
        text = chunk["text"]

        context_parts.append(
            f"[Page {page_number}]\n{text}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document question-answering assistant.

Your job is to answer the user's question using ONLY
the provided document context.

Rules:
- Use only information present in the document context.
- Do not invent or assume information.
- If the answer is not available in the context, clearly say:
  "The answer is not available in the provided document."
- Keep the answer clear and concise.
- Mention the page number that supports your answer.
- Do not mention information from outside the document.

Document Context:
{context}

User Question:
{question}

Answer:
"""

    try:

        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=prompt
        )

        return interaction.output_text

    except Exception as e:

        return f"API Error: {str(e)}"