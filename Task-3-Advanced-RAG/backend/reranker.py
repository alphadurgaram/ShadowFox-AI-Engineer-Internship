from sentence_transformers import CrossEncoder


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker = None


def get_reranker():
    global reranker

    if reranker is None:
        print("Loading reranker model...")

        reranker = CrossEncoder(
            RERANKER_MODEL
        )

        print("Reranker model loaded successfully.")

    return reranker


def rerank_chunks(
    question,
    chunks,
    top_k=3
):

    if not chunks:
        return []

    model = get_reranker()

    pairs = []

    for chunk in chunks:

        pairs.append(
            [
                question,
                chunk["text"]
            ]
        )

    scores = model.predict(pairs)

    ranked_chunks = []

    for chunk, score in zip(
        chunks,
        scores
    ):

        ranked_chunks.append(
            {
                "page": chunk["page"],
                "text": chunk["text"],
                "score": float(score)
            }
        )

    ranked_chunks.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return ranked_chunks[:top_k]