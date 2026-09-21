from pathlib import Path

from rag_engine import (
    load_document,
    chunk_text,
    create_vector_store,
    search_similar_chunks
)


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

query = input("\nAsk a question about the document: ")

results = search_similar_chunks(
    query,
    chunks,
    index,
    top_k=3
)

print("\nRelevant document sections:")

for i, result in enumerate(results, start=1):

    print(f"\n--- Result {i} ---")
    print(f"Page: {result['page']}")
    print(result["text"])