import streamlit as st
from pathlib import Path

from rag_engine import (
    load_document,
    chunk_text,
    create_vector_store,
    load_vector_store,
    search_similar_chunks
)

from ai_engine import generate_answer


st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)


st.title("📚 RAG Document Q&A Assistant")

st.write(
    "Upload a PDF or TXT document and ask questions based on its content."
)

st.divider()


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None


# --------------------------------------------------
# File Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF or TXT document",
    type=["pdf", "txt"]
)


if uploaded_file is not None:

    # Process only if a new document is uploaded
    if st.session_state.document_name != uploaded_file.name:

        documents_folder = Path("documents")
        documents_folder.mkdir(exist_ok=True)

        file_path = documents_folder / uploaded_file.name

        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        with st.spinner("Processing document..."):

            try:

                # Load document page by page
                pages = load_document(file_path)

                if not pages:
                    st.error(
                        "Could not extract text from the document."
                    )
                    st.stop()

                # Create chunks with page numbers
                chunks = chunk_text(pages)

                if not chunks:
                    st.error(
                        "No usable text chunks were created."
                    )
                    st.stop()

                # Create and save FAISS vector store
                vector_store = create_vector_store(chunks)

                # Save data in session state
                st.session_state.chunks = chunks
                st.session_state.vector_store = vector_store
                st.session_state.document_name = uploaded_file.name

                st.success(
                    f"Document processed successfully! "
                    f"{len(pages)} pages and "
                    f"{len(chunks)} chunks created."
                )

            except Exception as e:

                st.error(
                    f"Error processing document: {str(e)}"
                )


# --------------------------------------------------
# Load Saved Vector Store
# --------------------------------------------------

if (
    st.session_state.vector_store is None
    and st.session_state.chunks is None
):

    saved_index, saved_chunks = load_vector_store()

    if saved_index is not None and saved_chunks is not None:

        st.session_state.vector_store = saved_index
        st.session_state.chunks = saved_chunks
        st.session_state.document_name = "Saved Document"

        st.success(
            "Saved vector store loaded successfully."
        )


# --------------------------------------------------
# Question Answering
# --------------------------------------------------

if st.session_state.vector_store is not None:

    st.subheader("❓ Ask a Question")

    question = st.text_input(
        "Enter your question",
        placeholder="What is machine learning?"
    )

    if st.button(
        "🔍 Ask Question",
        use_container_width=True
    ):

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            with st.spinner(
                "Searching document and generating answer..."
            ):

                try:

                    # Retrieve relevant chunks
                    relevant_chunks = search_similar_chunks(
                        question,
                        st.session_state.chunks,
                        st.session_state.vector_store,
                        top_k=3
                    )

                    if not relevant_chunks:

                        st.warning(
                            "No relevant information was found."
                        )
                        st.stop()

                    # Generate grounded answer
                    answer = generate_answer(
                        question,
                        relevant_chunks
                    )

                    # Display answer
                    st.subheader("🤖 Answer")

                    if answer.startswith("API Error:"):

                        st.error(answer)

                    else:

                        st.markdown(answer)

                    # Display sources
                    st.subheader("📚 Retrieved Sources")

                    for i, chunk in enumerate(
                        relevant_chunks,
                        start=1
                    ):

                        page_number = chunk["page"]
                        text = chunk["text"]

                        with st.expander(
                            f"Source {i} — Page {page_number}"
                        ):

                            st.caption(
                                f"📄 Source Page: {page_number}"
                            )

                            st.write(text)

                except Exception as e:

                    st.error(
                        f"Error generating answer: {str(e)}"
                    )


else:

    st.info(
        "Please upload a PDF or TXT document to begin."
    )