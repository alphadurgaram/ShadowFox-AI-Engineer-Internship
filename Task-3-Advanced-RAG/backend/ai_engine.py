import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is not configured."
    )


client = genai.Client(
    api_key=api_key
)


def generate_answer(
    question,
    relevant_chunks
):

    if not relevant_chunks:
        return (
            "The answer is not available "
            "in the provided document."
        )

    context_parts = []

    for chunk in relevant_chunks:

        context_parts.append(
            f"[Page {chunk['page']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY
the provided document context.

Rules:
- Use only information from the context.
- Do not invent or assume information.
- If the answer is not available, say:
  "The answer is not available in the provided document."
- Keep the answer clear and concise.
- Mention the supporting page number.
- Do not use outside knowledge.

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


def check_groundedness(
    answer,
    relevant_chunks
):

    if not answer or not relevant_chunks:
        return False

    context_parts = []

    for chunk in relevant_chunks:

        context_parts.append(
            f"[Page {chunk['page']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are a groundedness verifier.

Your task is to determine whether the answer
is fully supported by the provided document context.

Rules:
- Compare the answer only against the provided context.
- Do not use outside knowledge.
- If the answer contains information not supported
  by the context, return FALSE.
- If the answer is fully supported by the context,
  return TRUE.
- Return ONLY one word:
  TRUE
  or
  FALSE

Document Context:
{context}

Answer:
{answer}

Verdict:
"""

    try:

        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=prompt
        )

        verdict = (
            interaction.output_text
            .strip()
            .upper()
        )

        return verdict == "TRUE"

    except Exception as e:

        print(
            f"Groundedness verification error: {e}"
        )

        return False