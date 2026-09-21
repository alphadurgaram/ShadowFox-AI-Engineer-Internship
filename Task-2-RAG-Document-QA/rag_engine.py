from pathlib import Path
import json

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Paths
# --------------------------------------------------

VECTOR_STORE_FOLDER = Path("vector_store")
INDEX_FILE = VECTOR_STORE_FOLDER / "document.index"
CHUNKS_FILE = VECTOR_STORE_FOLDER / "chunks.json"


# --------------------------------------------------
# Embedding Model
# --------------------------------------------------

embedding_model = None


def get_embedding_model():
    global embedding_model

    if embedding_model is None:
        print("Loading embedding model...")

        embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Embedding model loaded successfully.")

    return embedding_model


# --------------------------------------------------
# Load Document
# --------------------------------------------------

def load_document(file_path):
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".pdf":

        reader = PdfReader(file_path)

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            page_text = page.extract_text()

            if page_text:
                pages.append({
                    "page": page_number,
                    "text": page_text
                })

        return pages

    elif file_path.suffix.lower() == ".txt":

        text = file_path.read_text(
            encoding="utf-8"
        )

        return [
            {
                "page": 1,
                "text": text
            }
        ]

    else:

        raise ValueError(
            "Only PDF and TXT files are supported."
        )


# --------------------------------------------------
# Chunk Text
# --------------------------------------------------

def chunk_text(
    pages,
    chunk_size=500,
    overlap=50
):

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]
        text = page_data["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end].strip()

            if chunk:

                chunks.append({
                    "page": page_number,
                    "text": chunk
                })

            start += chunk_size - overlap

    return chunks


# --------------------------------------------------
# Create Vector Store
# --------------------------------------------------

def create_vector_store(chunks):

    if not chunks:
        raise ValueError(
            "No chunks available to create vector store."
        )

    model = get_embedding_model()

    print("Creating embeddings...")

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(texts)

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    print("Vector store created successfully.")

    # Create vector_store folder
    VECTOR_STORE_FOLDER.mkdir(
        exist_ok=True
    )

    # Save FAISS index
    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    # Save chunks and page metadata
    with open(
        CHUNKS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"FAISS index saved to: {INDEX_FILE}"
    )

    print(
        f"Chunks saved to: {CHUNKS_FILE}"
    )

    return index


# --------------------------------------------------
# Load Existing Vector Store
# --------------------------------------------------

def load_vector_store():

    if not INDEX_FILE.exists():
        return None, None

    if not CHUNKS_FILE.exists():
        return None, None

    print("Loading saved vector store...")

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    print("Saved vector store loaded successfully.")

    return index, chunks


# --------------------------------------------------
# Similarity Search
# --------------------------------------------------

def search_similar_chunks(
    query,
    chunks,
    index,
    top_k=3
):

    if not chunks or index is None:
        return []

    model = get_embedding_model()

    query_embedding = model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    actual_k = min(
        top_k,
        len(chunks)
    )

    distances, indices = index.search(
        query_embedding,
        actual_k
    )

    results = []

    for index_position in indices[0]:

        if index_position < len(chunks):

            results.append(
                chunks[index_position]
            )

    return results