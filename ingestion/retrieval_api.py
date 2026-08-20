import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from reranker import rerank


# ============================================================
# Configuration
# ============================================================

QDRANT_PATH = "C:/RAG-Goa/qdrant"
COLLECTION = "msmarco_hindi"

VECTOR_SIZE = 384

DEFAULT_TOP_K = 5

CANDIDATE_MULTIPLIER = 3
MIN_CANDIDATES = 15

RERANK_TOP_K = 5

RERANKER_NAME = (
    "cross-encoder/"
    "mmarco-mMiniLMv2-L12-H384-v1"
)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="RAG-Goa Retrieval Service",
    description="Qdrant retrieval with CrossEncoder reranking",
    version="2.2.0"
)


# ============================================================
# Qdrant
# ============================================================

try:

    client = QdrantClient(
        path=QDRANT_PATH
    )

except Exception as exc:

    raise RuntimeError(
        f"Failed to initialize Qdrant: {exc}"
    )


# ============================================================
# Request model
# ============================================================

class RetrieveRequest(BaseModel):

    query_embedding: list[float]

    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        le=50
    )

    min_score: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0
    )

    query: str = ""


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "collection": COLLECTION,
        "vector_size": VECTOR_SIZE,
        "candidate_limit": max(
            DEFAULT_TOP_K * CANDIDATE_MULTIPLIER,
            MIN_CANDIDATES
        ),
        "reranker": RERANKER_NAME
    }


# ============================================================
# Retrieval
# ============================================================

@app.post("/retrieve")
def retrieve(request: RetrieveRequest):

    total_start = time.perf_counter()

    # ========================================================
    # STEP 1 — Validate embedding
    # ========================================================

    if len(request.query_embedding) != VECTOR_SIZE:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Embedding must contain exactly "
                f"{VECTOR_SIZE} values."
            )
        )

    # ========================================================
    # STEP 2 — Clean query
    # ========================================================

    query = request.query.strip()

    # ========================================================
    # STEP 3 — Calculate candidate pool
    # ========================================================

    candidate_limit = max(
        request.top_k * CANDIDATE_MULTIPLIER,
        MIN_CANDIDATES
    )

    # ========================================================
    # STEP 4 — Qdrant retrieval
    # ========================================================

    qdrant_start = time.perf_counter()

    try:

        results = client.query_points(
            collection_name=COLLECTION,
            query=request.query_embedding,
            limit=candidate_limit
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Qdrant retrieval failed: "
                f"{exc}"
            )
        )

    qdrant_ms = (
        time.perf_counter() - qdrant_start
    ) * 1000

    # ========================================================
    # STEP 5 — Build unique candidates
    # ========================================================

    dedup_start = time.perf_counter()

    candidates = []

    seen_text = set()

    raw_points = results.points

    for point in raw_points:

        score = float(
            point.score
        )

        # ----------------------------------------------------
        # Similarity threshold
        # ----------------------------------------------------

        if score < request.min_score:
            continue

        payload = point.payload or {}

        text = payload.get("text")

        if not text:
            continue

        text = str(text).strip()

        if not text:
            continue

        # ----------------------------------------------------
        # Duplicate detection
        # ----------------------------------------------------

        normalized_text = (
            " ".join(
                text.split()
            ).lower()
        )

        if normalized_text in seen_text:
            continue

        seen_text.add(
            normalized_text
        )

        # ----------------------------------------------------
        # Candidate object
        # ----------------------------------------------------

        candidates.append({

            "score": score,

            "text": text,

            "source": payload.get(
                "source"
            ),

            "query": payload.get(
                "query"
            ),

            "language": payload.get(
                "language"
            ),

            "passage_index": payload.get(
                "passage_index"
            ),

            "is_selected": payload.get(
                "is_selected"
            ),

            "chunk_strategy": payload.get(
                "chunk_strategy"
            )
        })

    dedup_ms = (
        time.perf_counter() - dedup_start
    ) * 1000

    # ========================================================
    # STEP 6 — No candidates
    # ========================================================

    if not candidates:

        total_ms = (
            time.perf_counter() - total_start
        ) * 1000

        return {

            "results": [],

            "count": 0,

            "candidate_count":
                len(raw_points),

            "unique_count": 0,

            "reranked": False,

            "reranker":
                RERANKER_NAME,

            "latency": {

                "qdrant_ms":
                    round(qdrant_ms, 2),

                "dedup_ms":
                    round(dedup_ms, 2),

                "rerank_ms":
                    0.0,

                "total_ms":
                    round(total_ms, 2)
            }
        }

    # ========================================================
    # STEP 7 — CrossEncoder reranking
    # ========================================================

    rerank_ms = 0.0
    reranked = False

    if query:

        rerank_start = time.perf_counter()

        try:

            reranked_results = rerank(

                query=query,

                candidates=candidates,

                top_k=min(
                    request.top_k,
                    RERANK_TOP_K
                )
            )

        except Exception as exc:

            raise HTTPException(
                status_code=500,
                detail=(
                    "CrossEncoder reranking failed: "
                    f"{exc}"
                )
            )

        rerank_ms = (
            time.perf_counter()
            - rerank_start
        ) * 1000

        reranked = True

    else:

        candidates.sort(
            key=lambda item:
                item["score"],
            reverse=True
        )

        reranked_results = candidates[
            :request.top_k
        ]

    # ========================================================
    # STEP 8 — Normalize results
    # ========================================================

    normalize_start = time.perf_counter()

    final_results = []

    for item in reranked_results:

        result = dict(item)

        if reranked:

            if "rerank_score" not in result:

                result["rerank_score"] = None

        else:

            result["rerank_score"] = None

        final_results.append(
            result
        )

    normalize_ms = (
        time.perf_counter()
        - normalize_start
    ) * 1000

    # ========================================================
    # STEP 9 — Total latency
    # ========================================================

    total_ms = (
        time.perf_counter()
        - total_start
    ) * 1000

    # ========================================================
    # STEP 10 — Response
    # ========================================================

    return {

        "results":
            final_results,

        "count":
            len(final_results),

        "candidate_count":
            len(raw_points),

        "unique_count":
            len(candidates),

        "reranked":
            reranked,

        "reranker":
            RERANKER_NAME,

        "latency": {

            "qdrant_ms":
                round(qdrant_ms, 2),

            "dedup_ms":
                round(dedup_ms, 2),

            "rerank_ms":
                round(rerank_ms, 2),

            "normalize_ms":
                round(normalize_ms, 2),

            "total_ms":
                round(total_ms, 2)
        }
    }


# ============================================================
# Shutdown
# ============================================================

@app.on_event("shutdown")
def shutdown():

    try:

        client.close()

    except Exception:

        pass