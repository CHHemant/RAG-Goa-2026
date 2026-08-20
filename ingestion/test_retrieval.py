import requests


# ============================================================
# Configuration
# ============================================================

EMBED_URL = "http://127.0.0.1:8000/embed"
RETRIEVAL_URL = "http://127.0.0.1:8001/retrieve"

QUERY = "मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?"

TOP_K = 5
MIN_SCORE = 0.0


# ============================================================
# 1. Generate QUERY embedding
# ============================================================

print("Generating query embedding...")

embedding_response = requests.post(
    EMBED_URL,
    json={
        "text": QUERY,
        "input_type": "query"
    },
    timeout=60
)

embedding_response.raise_for_status()

embedding_data = embedding_response.json()

query_embedding = embedding_data["embedding"]


# ============================================================
# Validate embedding
# ============================================================

if len(query_embedding) != 384:
    raise ValueError(
        f"Expected 384-dimensional embedding, "
        f"got {len(query_embedding)}"
    )

print("Embedding dimension:", len(query_embedding))


# ============================================================
# 2. Send query + embedding to Retrieval API
# ============================================================

print("Sending query to retrieval service...")

retrieval_response = requests.post(
    RETRIEVAL_URL,
    json={
        "query_embedding": query_embedding,

        # IMPORTANT:
        # The CrossEncoder reranker needs
        # the original query text.
        "query": QUERY,

        "top_k": TOP_K,
        "min_score": MIN_SCORE
    },
    timeout=120
)

retrieval_response.raise_for_status()

data = retrieval_response.json()


# ============================================================
# 3. Display query
# ============================================================

print()
print("QUERY:")
print(QUERY)

print()
print("=" * 80)


# ============================================================
# 4. Display retrieval + reranking results
# ============================================================

for i, result in enumerate(
    data.get("results", []),
    start=1
):

    print(f"\n--- Result {i} ---")

    print(
        "Qdrant Score:",
        result.get("score")
    )

    print(
        "Rerank Score:",
        result.get("rerank_score", "N/A")
    )

    print(
        "Text:",
        result.get("text")
    )

    print(
        "Source:",
        result.get("source")
    )

    print(
        "Language:",
        result.get("language")
    )

    print(
        "Passage Index:",
        result.get("passage_index")
    )

    print(
        "Selected:",
        result.get("is_selected")
    )

    print(
        "Chunk:",
        result.get("chunk_strategy")
    )


# ============================================================
# 5. Retrieval statistics
# ============================================================

print()
print("=" * 80)

print(
    "Results:",
    data.get("count", 0)
)

print(
    "Candidates retrieved:",
    data.get("candidate_count", 0)
)

print(
    "Unique candidates:",
    data.get("unique_count", 0)
)

print(
    "Reranking enabled:",
    data.get("reranked", False)
)

print("=" * 80)