def get_prompt(feature, content):

    if feature == "Summarize Notes":
        return f"""
You are an AI study assistant.

Task:
Summarize the following study notes.

Requirements:
- Keep all important concepts.
- Remove unnecessary repetition.
- Use simple and clear language.
- Organize the summary using headings and bullet points.
- Do not add information that is not present in the notes.

Study Notes:
{content}
"""

    elif feature == "Generate Quiz":
        return f"""
You are an AI study assistant.

Task:
Create a quiz from the following study material.

Requirements:
- Generate 5 multiple-choice questions.
- Each question must have 4 options.
- Clearly mention the correct answer.
- Questions must be based only on the provided study material.
- Include a mixture of conceptual and factual questions.

Study Material:
{content}
"""

    elif feature == "Improve Answer":
        return f"""
You are an AI study assistant.

Task:
Improve the student's answer given below.

Requirements:
- Preserve the original meaning.
- Correct grammar and unclear wording.
- Make the answer more structured and understandable.
- Do not add unsupported facts.
- Provide the improved answer first.
- Then briefly explain what was improved.

Student Answer:
{content}
"""

    elif feature == "Explain Concept":
        return f"""
You are an AI study assistant.

Task:
Explain the following concept to a student.

Requirements:
- Start with a simple definition.
- Explain it step by step.
- Give a simple example.
- Use easy-to-understand language.
- Avoid unnecessary technical complexity.

Concept:
{content}
"""

    return ""