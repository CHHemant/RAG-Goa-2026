import requests

from rag_generator import generate_answer


# ============================================================
# Configuration
# ============================================================

EMBED_URL = "http://127.0.0.1:8000/embed"

RETRIEVAL_URL = "http://127.0.0.1:8001/retrieve"

QUERY = "मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?"


# ============================================================
# 1. Generate query embedding
# ============================================================

print("Generating embedding...")

embedding_response = requests.post(

    EMBED_URL,

    json={
        "text": QUERY,
        "input_type": "query"
    },

    timeout=60
)

embedding_response.raise_for_status()

query_embedding = embedding_response.json()["embedding"]


# ============================================================
# 2. Retrieve + rerank
# ============================================================

print("Retrieving context...")

retrieval_response = requests.post(

    RETRIEVAL_URL,

    json={

        "query_embedding": query_embedding,

        "query": QUERY,

        "top_k": 5,

        "min_score": 0.0

    },

    timeout=120
)

retrieval_response.raise_for_status()

retrieval_data = retrieval_response.json()

contexts = retrieval_data.get(
    "results",
    []
)


print(
    "Retrieved contexts:",
    len(contexts)
)


# ============================================================
# 3. Generate final answer
# ============================================================

print("Generating answer with OpenRouter...")

result = generate_answer(
    query=QUERY,
    contexts=contexts
)


# ============================================================
# 4. Display
# ============================================================

print()
print("=" * 80)

print("QUERY:")
print(QUERY)

print()
print("ANSWER:")
print(result["answer"])

print()
print("MODEL:")
print(result["model"])

print()
print("SOURCES:")

for source in result["sources"]:

    print(
        source
    )

print("=" * 80)