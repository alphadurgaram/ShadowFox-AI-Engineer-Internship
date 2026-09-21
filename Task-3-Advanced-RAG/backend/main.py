from pathlib import Path
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException

from backend.models import QuestionRequest

from backend.rag_engine import (
    load_document,
    chunk_text,
    create_vector_store,
    load_vector_store
)

from backend.ai_engine import generate_answer
from backend.graph import build_graph


app = FastAPI(
    title="Advanced RAG API",
    description="Document Question Answering API",
    version="1.0.0"
)


DOCUMENTS_FOLDER = Path("documents")
DOCUMENTS_FOLDER.mkdir(exist_ok=True)


vector_store = None
all_chunks = None
rag_graph = None


@app.get("/")
def home():

    return {
        "message": "Advanced RAG API is running"
    }


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    global vector_store
    global all_chunks
    global rag_graph

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in [".pdf", ".txt"]:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported."
        )

    file_path = (
        DOCUMENTS_FOLDER /
        file.filename
    )

    try:

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        pages = load_document(
            file_path
        )

        if not pages:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from document."
            )

        chunks = chunk_text(
            pages
        )

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No usable text chunks found."
            )

        index = create_vector_store(
            chunks
        )

        vector_store = index
        all_chunks = chunks

        rag_graph = build_graph(
            vector_store,
            all_chunks,
            generate_answer
        )

        return {
            "message": "Document processed successfully.",
            "filename": file.filename,
            "pages": len(pages),
            "chunks": len(chunks)
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )


@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    global vector_store
    global all_chunks
    global rag_graph

    if vector_store is None:

        vector_store, all_chunks = (
            load_vector_store()
        )

        if vector_store is not None:

            rag_graph = build_graph(
                vector_store,
                all_chunks,
                generate_answer
            )

    if vector_store is None or not all_chunks:

        raise HTTPException(
            status_code=400,
            detail="Please upload a document first."
        )

    try:

        result = rag_graph.invoke(
            {
                "question": request.question
            }
        )

        sources = []

        for chunk in result.get(
            "ranked_chunks",
            []
        ):

            sources.append({
                "page": chunk["page"],
                "text": chunk["text"]
            })

        return {
            "answer": result.get(
                "answer",
                ""
            ),
            "sources": sources,
            "grounded": result.get(
                "grounded",
                False
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Question processing error: {str(e)}"
        )