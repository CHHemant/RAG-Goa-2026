from sentence_transformers import CrossEncoder


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

DEVICE = "cpu"


# ============================================================
# Load reranker once
# ============================================================

print(f"Loading reranker: {MODEL_NAME}")
print(f"Device: {DEVICE}")

model = CrossEncoder(
    MODEL_NAME,
    device=DEVICE
)

print("Reranker loaded successfully.")


# ============================================================
# Rerank candidates
# ============================================================

def rerank(query: str, candidates: list[dict], top_k: int = 3):

    if not query.strip():
        return []

    if not candidates:
        return []

    pairs = [
        [query, candidate["text"]]
        for candidate in candidates
    ]

    scores = model.predict(
        pairs,
        show_progress_bar=False
    )

    reranked = []

    for candidate, score in zip(candidates, scores):

        result = dict(candidate)

        result["rerank_score"] = float(score)

        reranked.append(result)

    reranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return reranked[:top_k]