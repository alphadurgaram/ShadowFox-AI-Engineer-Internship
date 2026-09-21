from pathlib import Path

from rag_engine import (
    load_document,
    chunk_text,
    create_vector_store,
    search_similar_chunks
)

from ai_engine import generate_answer


documents_folder = Path("documents")

files = list(documents_folder.glob("*.pdf"))
files += list(documents_folder.glob("*.txt"))

if not files:
    print("No PDF or TXT file found in the documents folder.")
    exit()

file_path = files[0]

print("Loading document:", file_path.name)

pages = load_document(file_path)

if not pages:
    print("Could not extract text from the document.")
    exit()

print("Document loaded successfully.")
print("Pages loaded:", len(pages))

chunks = chunk_text(pages)

print("Number of chunks:", len(chunks))

index = create_vector_store(chunks)

print("Vector store created successfully.")

question = input("\nAsk a question about the document: ")

relevant_chunks = search_similar_chunks(
    question,
    chunks,
    index,
    top_k=3
)

print("\nGenerating answer...")

answer = generate_answer(
    question,
    relevant_chunks
)

print("\n===== AI ANSWER =====")
print(answer)

print("\n===== RETRIEVED SOURCES =====")

for i, chunk in enumerate(relevant_chunks, start=1):

    print(f"\n--- Source {i} | Page {chunk['page']} ---")
    print(chunk["text"])