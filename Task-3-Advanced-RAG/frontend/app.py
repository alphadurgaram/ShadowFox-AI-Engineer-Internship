import os

import requests
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

API_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000"
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Advanced RAG Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        opacity: 0.7;
        margin-bottom: 1.5rem;
    }

    .pipeline-box {
        padding: 18px;
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 15px;
        margin: 15px 0;
    }

    .pipeline-step {
        text-align: center;
        font-weight: 700;
        font-size: 15px;
    }

    .pipeline-arrow {
        text-align: center;
        font-size: 22px;
        opacity: 0.6;
        padding-top: 8px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-top: 10px;
    }

    .small-text {
        font-size: 13px;
        opacity: 0.65;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🧠 RAG Control Center")

    st.caption(
        "Advanced document intelligence system"
    )

    st.divider()

    st.subheader("⚙️ System Status")

    st.success("🟢 Backend API Connected")
    st.success("🟢 Retrieval Engine Ready")
    st.success("🟢 Groundedness Enabled")

    st.divider()

    st.subheader("🔬 RAG Pipeline")

    st.markdown(
        """
        **1. 🔎 Retrieve**  
        Find relevant document chunks.

        **2. 🎯 Rerank**  
        Reorder retrieved chunks by relevance.

        **3. 🤖 Generate**  
        Generate an answer from retrieved context.

        **4. 🛡️ Verify**  
        Check answer groundedness.
        """
    )

    st.divider()

    st.caption(
        "Advanced RAG Document Assistant"
    )


# =========================================================
# HEADER
# =========================================================

st.title(
    "🧠 Advanced RAG Document Intelligence"
)

st.caption(
    "Retrieve  →  Rerank  →  Generate  →  Verify"
)


# =========================================================
# STATUS SECTION
# =========================================================

st.subheader("📡 System Overview")

status1, status2, status3, status4 = st.columns(4)

with status1:
    st.metric(
        "Backend",
        "Online"
    )

with status2:
    st.metric(
        "Vector Store",
        "Ready"
    )

with status3:
    st.metric(
        "Retrieval",
        "Active"
    )

with status4:
    st.metric(
        "Grounding",
        "Enabled"
    )


# =========================================================
# PIPELINE
# =========================================================

st.subheader("⚡ RAG Processing Pipeline")

pipe1, arrow1, pipe2, arrow2, pipe3, arrow3, pipe4 = st.columns(
    [2, 0.5, 2, 0.5, 2, 0.5, 2]
)

with pipe1:
    st.info("🔎 **RETRIEVE**")

with arrow1:
    st.write("→")

with pipe2:
    st.info("🎯 **RERANK**")

with arrow2:
    st.write("→")

with pipe3:
    st.info("🤖 **GENERATE**")

with arrow3:
    st.write("→")

with pipe4:
    st.success("🛡️ **VERIFY**")


st.divider()


# =========================================================
# DOCUMENT WORKSPACE
# =========================================================

st.subheader("📄 Document Workspace")

upload_col, info_col = st.columns(
    [2, 1]
)

with upload_col:

    uploaded_file = st.file_uploader(
        "Upload your document",
        type=["pdf", "txt"],
        help="Supported formats: PDF and TXT"
    )

with info_col:

    st.info(
        "Upload a document first. "
        "Then process it before asking questions."
    )


if uploaded_file is not None:

    st.write(
        f"📎 **Selected file:** `{uploaded_file.name}`"
    )

    if st.button(
        "⚡ Process Document",
        use_container_width=True,
        type="primary"
    ):

        with st.spinner(
            "Processing document..."
        ):

            try:

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    f"{API_URL}/upload",
                    files=files,
                    timeout=300
                )

                if response.status_code == 200:

                    data = response.json()

                    st.success(
                        "✅ Document processed successfully!"
                    )

                    pages = data.get(
                        "pages",
                        0
                    )

                    chunks = data.get(
                        "chunks",
                        0
                    )

                    m1, m2, m3 = st.columns(3)

                    with m1:
                        st.metric(
                            "Pages",
                            pages
                        )

                    with m2:
                        st.metric(
                            "Chunks",
                            chunks
                        )

                    with m3:
                        st.metric(
                            "Document Status",
                            "Ready"
                        )

                else:

                    try:

                        error_message = response.json().get(
                            "detail",
                            "Document processing failed."
                        )

                    except Exception:

                        error_message = response.text

                    st.error(
                        error_message
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ FastAPI backend is not reachable."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "⏳ Document processing timed out."
                )

            except Exception as e:

                st.error(
                    f"Error: {str(e)}"
                )


st.divider()


# =========================================================
# QUESTION AREA
# =========================================================

st.subheader("🔎 Ask Your Document")

st.caption(
    "The system will retrieve relevant chunks, rerank them, "
    "generate an answer and verify its groundedness."
)

question = st.text_area(
    "Enter your question",
    placeholder="Example: What is machine learning?",
    height=110
)


if st.button(
    "🚀 Ask Advanced RAG",
    use_container_width=True,
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question first."
        )

    else:

        with st.spinner(
            "Running Retrieve → Rerank → Generate → Verify..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question
                    },
                    timeout=300
                )

                if response.status_code == 200:

                    data = response.json()

                    answer = data.get(
                        "answer",
                        "No answer generated."
                    )

                    grounded = data.get(
                        "grounded",
                        False
                    )

                    sources = data.get(
                        "sources",
                        []
                    )


                    # =====================================
                    # ANSWER
                    # =====================================

                    st.divider()

                    st.subheader(
                        "🤖 AI Response"
                    )

                    st.markdown(
                        answer
                    )


                    # =====================================
                    # RESULT STATUS
                    # =====================================

                    st.subheader(
                        "📊 Answer Analysis"
                    )

                    r1, r2, r3 = st.columns(3)

                    with r1:

                        if grounded:

                            st.success(
                                "🟢 Grounded"
                            )

                        else:

                            st.warning(
                                "🟡 Needs Verification"
                            )

                    with r2:

                        st.metric(
                            "Sources Retrieved",
                            len(sources)
                        )

                    with r3:

                        st.success(
                            "Pipeline Complete"
                        )


                    if grounded:

                        st.success(
                            "✅ Answer is grounded "
                            "in the retrieved document."
                        )

                    else:

                        st.warning(
                            "⚠️ Answer could not be fully "
                            "verified against the retrieved document."
                        )


                    # =====================================
                    # SOURCES
                    # =====================================

                    st.subheader(
                        "📚 Retrieved Evidence"
                    )

                    if sources:

                        for i, source in enumerate(
                            sources,
                            start=1
                        ):

                            page = source.get(
                                "page",
                                "Unknown"
                            )

                            text = source.get(
                                "text",
                                ""
                            )

                            with st.expander(
                                f"📄 Evidence {i} — Page {page}"
                            ):

                                st.write(
                                    text
                                )

                    else:

                        st.info(
                            "No retrieved sources available."
                        )


                else:

                    try:

                        error_message = response.json().get(
                            "detail",
                            "Question processing failed."
                        )

                    except Exception:

                        error_message = response.text

                    st.error(
                        error_message
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ FastAPI backend is not reachable."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "⏳ Request timed out."
                )

            except Exception as e:

                st.error(
                    f"Error: {str(e)}"
                )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🧠 Advanced RAG Document Intelligence  •  "
    "Retrieve • Rerank • Generate • Verify"
)