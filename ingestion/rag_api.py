import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_generator import generate_answer


# ============================================================
# Load project environment
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


# ============================================================
# Configuration
# ============================================================

EMBED_URL = "http://127.0.0.1:8000/embed"
RETRIEVAL_URL = "http://127.0.0.1:8001/retrieve"

EMBED_TIMEOUT = 60
RETRIEVAL_TIMEOUT = 120

EXPECTED_EMBEDDING_DIMENSION = 384

DEFAULT_TOP_K = 5

RERANK_THRESHOLD = -1.0

NO_CONTEXT_ANSWER = (
    "दिए गए संदर्भ में इस प्रश्न का "
    "पर्याप्त उत्तर उपलब्ध नहीं है।"
)

RERANKER_NAME = (
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="RAG-Goa API",
    description="Hindi Retrieval-Augmented Generation API",
    version="1.3.0"
)


# ============================================================
# Request schema
# ============================================================

class AskRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=1,
        description="User question"
    )

    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        le=20,
        description="Number of final retrieved contexts"
    )

    min_score: float = Field(
        default=0.0,
        description="Minimum Qdrant similarity score"
    )


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "RAG-Goa",
        "embedding_service": EMBED_URL,
        "retrieval_service": RETRIEVAL_URL,
        "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
        "rerank_threshold": RERANK_THRESHOLD,
        "reranker": RERANKER_NAME
    }


# ============================================================
# Helper — safe integer conversion
# ============================================================

def safe_int(
    value,
    default=0
):

    try:
        return int(value)

    except (
        TypeError,
        ValueError
    ):

        return default


# ============================================================
# Helper — safe float conversion
# ============================================================

def safe_float(value):

    try:
        return float(value)

    except (
        TypeError,
        ValueError
    ):

        return None


# ============================================================
# Helper — round latency
# ============================================================

def latency_ms(start_time):

    return round(
        (time.perf_counter() - start_time) * 1000,
        2
    )


# ============================================================
# Helper — no-context response
# ============================================================

def build_no_context_response(
    query: str,
    candidate_count: int,
    unique_count: int,
    returned: int,
    reranked: bool,
    reranker_name: str,
    guardrail: str | None = None,
    best_rerank_score: float | None = None,
    latency: dict | None = None
):

    retrieval = {
        "candidates": candidate_count,
        "unique_candidates": unique_count,
        "returned": returned,
        "reranked": reranked,
        "reranker": reranker_name
    }

    if guardrail is not None:
        retrieval["guardrail"] = guardrail

    if best_rerank_score is not None:
        retrieval["best_rerank_score"] = (
            best_rerank_score
        )

    response = {
        "query": query,
        "answer": NO_CONTEXT_ANSWER,
        "model": None,
        "grounded": False,
        "sources": [],
        "retrieval": retrieval
    }

    if latency is not None:
        response["latency"] = latency

    return response


# ============================================================
# Main RAG endpoint
# ============================================================

@app.post("/ask")
def ask(request: AskRequest):

    # ========================================================
    # Total request timer
    # ========================================================

    total_start = time.perf_counter()


    # ========================================================
    # STEP 0 — Validate query
    # ========================================================

    query = request.query.strip()

    if not query:

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )


    # ========================================================
    # STEP 1 — Generate query embedding
    # ========================================================

    embedding_start = time.perf_counter()

    try:

        embedding_response = requests.post(

            EMBED_URL,

            json={
                "text": query,
                "input_type": "query"
            },

            timeout=EMBED_TIMEOUT
        )

    except requests.Timeout:

        raise HTTPException(
            status_code=503,
            detail="Embedding service timed out."
        )

    except requests.RequestException as exc:

        raise HTTPException(
            status_code=503,
            detail=(
                "Embedding service unavailable: "
                f"{exc}"
            )
        )

    embedding_ms = latency_ms(
        embedding_start
    )


    # ========================================================
    # Check embedding HTTP status
    # ========================================================

    if not embedding_response.ok:

        raise HTTPException(
            status_code=502,
            detail=(
                "Embedding service returned HTTP "
                f"{embedding_response.status_code}: "
                f"{embedding_response.text[:500]}"
            )
        )


    # ========================================================
    # Parse embedding JSON
    # ========================================================

    try:

        embedding_data = (
            embedding_response.json()
        )

    except ValueError:

        raise HTTPException(
            status_code=502,
            detail=(
                "Embedding service returned invalid JSON."
            )
        )


    query_embedding = embedding_data.get(
        "embedding"
    )


    # ========================================================
    # Validate embedding
    # ========================================================

    if not isinstance(
        query_embedding,
        list
    ):

        raise HTTPException(
            status_code=502,
            detail=(
                "Embedding service returned "
                "an invalid embedding."
            )
        )


    if len(query_embedding) != (
        EXPECTED_EMBEDDING_DIMENSION
    ):

        raise HTTPException(
            status_code=502,
            detail=(
                "Invalid embedding dimension. "
                f"Expected "
                f"{EXPECTED_EMBEDDING_DIMENSION}, "
                f"got {len(query_embedding)}."
            )
        )


    # ========================================================
    # STEP 2 — Qdrant retrieval + reranking
    # ========================================================

    retrieval_start = time.perf_counter()

    try:

        retrieval_response = requests.post(

            RETRIEVAL_URL,

            json={

                "query_embedding":
                    query_embedding,

                "query":
                    query,

                "top_k":
                    request.top_k,

                "min_score":
                    request.min_score
            },

            timeout=RETRIEVAL_TIMEOUT
        )

    except requests.Timeout:

        raise HTTPException(
            status_code=503,
            detail="Retrieval service timed out."
        )

    except requests.RequestException as exc:

        raise HTTPException(
            status_code=503,
            detail=(
                "Retrieval service unavailable: "
                f"{exc}"
            )
        )

    retrieval_ms = latency_ms(
        retrieval_start
    )


    # ========================================================
    # Check retrieval HTTP status
    # ========================================================

    if not retrieval_response.ok:

        raise HTTPException(
            status_code=502,
            detail=(
                "Retrieval service returned HTTP "
                f"{retrieval_response.status_code}: "
                f"{retrieval_response.text[:1000]}"
            )
        )


    # ========================================================
    # Parse retrieval JSON
    # ========================================================

    try:

        retrieval_data = (
            retrieval_response.json()
        )

    except ValueError:

        raise HTTPException(
            status_code=502,
            detail=(
                "Retrieval service returned invalid JSON."
            )
        )


    # ========================================================
    # STEP 3 — Extract contexts
    # ========================================================

    contexts = retrieval_data.get(
        "results",
        []
    )


    if not isinstance(
        contexts,
        list
    ):

        raise HTTPException(
            status_code=502,
            detail=(
                "Retrieval service returned "
                "invalid results."
            )
        )


    # ========================================================
    # STEP 4 — Retrieval metadata
    # ========================================================

    candidate_count = safe_int(
        retrieval_data.get(
            "candidate_count",
            0
        )
    )

    unique_count = safe_int(
        retrieval_data.get(
            "unique_count",
            retrieval_data.get(
                "unique_candidates",
                0
            )
        )
    )

    reranked = bool(
        retrieval_data.get(
            "reranked",
            False
        )
    )

    reranker_name = retrieval_data.get(
        "reranker",
        RERANKER_NAME
    )


    # ========================================================
    # STEP 5 — No retrieval results
    # ========================================================

    if not contexts:

        total_ms = latency_ms(
            total_start
        )

        return build_no_context_response(

            query=query,

            candidate_count=
                candidate_count,

            unique_count=
                unique_count,

            returned=0,

            reranked=
                reranked,

            reranker_name=
                reranker_name,

            guardrail=
                "no_retrieval_results",

            latency={
                "embedding_ms":
                    embedding_ms,

                "retrieval_ms":
                    retrieval_ms,

                "generation_ms":
                    0.0,

                "total_ms":
                    total_ms
            }
        )


    # ========================================================
    # STEP 6 — Retrieval confidence guardrail
    # ========================================================

    rerank_scores = []

    for context in contexts:

        if not isinstance(
            context,
            dict
        ):
            continue

        score = safe_float(
            context.get(
                "rerank_score"
            )
        )

        if score is not None:
            rerank_scores.append(
                score
            )


    best_rerank_score = None

    if rerank_scores:

        best_rerank_score = max(
            rerank_scores
        )


    # ========================================================
    # Reject clearly irrelevant retrieval
    # ========================================================

    if (
        reranked
        and best_rerank_score is not None
        and best_rerank_score < RERANK_THRESHOLD
    ):

        total_ms = latency_ms(
            total_start
        )

        return build_no_context_response(

            query=query,

            candidate_count=
                candidate_count,

            unique_count=
                unique_count,

            returned=
                len(contexts),

            reranked=
                reranked,

            reranker_name=
                reranker_name,

            guardrail=
                "rejected_low_relevance",

            best_rerank_score=
                best_rerank_score,

            latency={
                "embedding_ms":
                    embedding_ms,

                "retrieval_ms":
                    retrieval_ms,

                "generation_ms":
                    0.0,

                "total_ms":
                    total_ms
            }
        )


    # ========================================================
    # STEP 7 — NVIDIA generation
    # ========================================================

    generation_start = time.perf_counter()

    try:

        generation_result = generate_answer(

            query=query,

            contexts=contexts
        )

    except Exception as exc:

        generation_ms = latency_ms(
            generation_start
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Generation failed: "
                f"{exc}"
            )
        )

    generation_ms = latency_ms(
        generation_start
    )


    # ========================================================
    # STEP 8 — Validate generation
    # ========================================================

    if not isinstance(
        generation_result,
        dict
    ):

        raise HTTPException(
            status_code=502,
            detail=(
                "Generation service returned "
                "an invalid response."
            )
        )


    answer = generation_result.get(
        "answer"
    )


    if (
        not isinstance(
            answer,
            str
        )
        or not answer.strip()
    ):

        raise HTTPException(
            status_code=502,
            detail=(
                "Generation service returned "
                "an empty answer."
            )
        )


    # ========================================================
    # STEP 9 — Sources
    # ========================================================

    sources = generation_result.get(
        "sources",
        []
    )


    if not isinstance(
        sources,
        list
    ):

        sources = []


    # ========================================================
    # STEP 10 — Preserve rerank scores
    # ========================================================

    for source in sources:

        if not isinstance(
            source,
            dict
        ):
            continue

        context_id = source.get(
            "context_id"
        )

        if context_id is None:
            continue

        try:

            context_index = (
                int(context_id) - 1
            )

        except (
            TypeError,
            ValueError
        ):

            continue

        if (
            context_index < 0
            or context_index >= len(contexts)
        ):
            continue

        context = contexts[
            context_index
        ]

        if (
            source.get(
                "rerank_score"
            ) is None
        ):

            source["rerank_score"] = (
                context.get(
                    "rerank_score"
                )
            )


    # ========================================================
    # STEP 11 — Total latency
    # ========================================================

    total_ms = latency_ms(
        total_start
    )


    # ========================================================
    # STEP 12 — Final response
    # ========================================================

    return {

        "query":
            query,

        "answer":
            answer.strip(),

        "model":
            generation_result.get(
                "model"
            ),

        "grounded":
            bool(
                generation_result.get(
                    "grounded",
                    True
                )
            ),

        "sources":
            sources,

        "retrieval": {

            "candidates":
                candidate_count,

            "unique_candidates":
                unique_count,

            "returned":
                len(contexts),

            "reranked":
                reranked,

            "reranker":
                reranker_name,

            "best_rerank_score":
                best_rerank_score,

            "guardrail":
                "passed"

        },

        "latency": {

            "embedding_ms":
                embedding_ms,

            "retrieval_ms":
                retrieval_ms,

            "generation_ms":
                generation_ms,

            "total_ms":
                total_ms
        }

    }