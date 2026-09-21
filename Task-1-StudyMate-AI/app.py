import streamlit as st

from ai_engine import generate_response
from prompts import get_prompt


# Page configuration
st.set_page_config(
    page_title="StudyMate AI",
    page_icon="🎓",
    layout="wide"
)


# App title
st.title("🎓 StudyMate AI")
st.write("Your AI-powered study assistant")


st.divider()


# Feature selection
feature = st.selectbox(
    "Choose a study utility",
    [
        "Summarize Notes",
        "Generate Quiz",
        "Improve Answer",
        "Explain Concept"
    ]
)


# User input
st.subheader("Enter your content")

content = st.text_area(
    "Paste your notes, answer, or concept here:",
    height=250,
    placeholder="Enter your study content..."
)


# Generate button
if st.button("✨ Generate", use_container_width=True):

    # Validate input
    if not content.strip():
        st.warning("Please enter some content first.")

    elif len(content.strip()) < 10:
        st.warning("Please enter at least a little more content.")

    else:

        with st.spinner("AI is generating your response..."):

            prompt = get_prompt(feature, content)

            result = generate_response(prompt)


        # Display result
        st.subheader("📚 AI Result")

        if result.startswith("API Error:"):
            st.error(result)
        else:
            st.markdown(result)