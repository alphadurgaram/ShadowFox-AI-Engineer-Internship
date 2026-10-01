from typing import TypedDict

from langgraph.graph import StateGraph, END

from backend.rag_engine import search_similar_chunks
from backend.reranker import rerank_chunks
from backend.ai_engine import check_groundedness


class RAGState(TypedDict, total=False):
    question: str
    chunks: list
    ranked_chunks: list
    answer: str
    grounded: bool


def retrieve_node(
    state: RAGState,
    vector_store,
    all_chunks
):
    question = state["question"]

    chunks = search_similar_chunks(
        question,
        all_chunks,
        vector_store,
        top_k=5
    )

    return {
        "chunks": chunks
    }


def rerank_node(
    state: RAGState
):
    ranked_chunks = rerank_chunks(
        state["question"],
        state.get("chunks", []),
        top_k=3
    )

    return {
        "ranked_chunks": ranked_chunks
    }


def generate_node(
    state: RAGState,
    generate_answer_function
):
    answer = generate_answer_function(
        state["question"],
        state.get("ranked_chunks", [])
    )

    return {
        "answer": answer
    }


def groundedness_node(
    state: RAGState
):
    answer = state.get(
        "answer",
        ""
    )

    ranked_chunks = state.get(
        "ranked_chunks",
        []
    )

    grounded = check_groundedness(
        answer,
        ranked_chunks
    )

    return {
        "grounded": grounded
    }


def build_graph(
    vector_store,
    all_chunks,
    generate_answer_function
):
    workflow = StateGraph(RAGState)

    workflow.add_node(
        "retrieve",
        lambda state: retrieve_node(
            state,
            vector_store,
            all_chunks
        )
    )

    workflow.add_node(
        "rerank",
        rerank_node
    )

    workflow.add_node(
        "generate",
        lambda state: generate_node(
            state,
            generate_answer_function
        )
    )

    workflow.add_node(
        "groundedness",
        groundedness_node
    )

    workflow.set_entry_point(
        "retrieve"
    )

    workflow.add_edge(
        "retrieve",
        "rerank"
    )

    workflow.add_edge(
        "rerank",
        "generate"
    )

    workflow.add_edge(
        "generate",
        "groundedness"
    )

    workflow.add_edge(
        "groundedness",
        END
    )

    return workflow.compile()