from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import time


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "intfloat/multilingual-e5-small"
DEVICE = "cpu"


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

print(f"Loading embedding model: {MODEL_NAME}")
print(f"Device: {DEVICE}")

model = SentenceTransformer(
    MODEL_NAME,
    device=DEVICE
)

print("Embedding model loaded successfully.")


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="RAG-Goa Embedding Service",
    version="1.1.0"
)


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class EmbeddingRequest(BaseModel):
    text: str
    input_type: str = "passage"


# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "dimension": 384
    }


# --------------------------------------------------
# Generate embedding
# --------------------------------------------------

@app.post("/embed")
def generate_embedding(request: EmbeddingRequest):

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )

    if request.input_type not in ["query", "passage"]:
        raise HTTPException(
            status_code=400,
            detail="input_type must be either 'query' or 'passage'."
        )

    started_at = time.perf_counter()

    try:

        # --------------------------------------------------
        # E5 instruction
        # --------------------------------------------------

        if request.input_type == "query":
            formatted_text = f"query: {text}"
        else:
            formatted_text = f"passage: {text}"

        # --------------------------------------------------
        # Generate embedding
        # --------------------------------------------------

        embedding = model.encode(
            formatted_text,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        latency_ms = round(
            (time.perf_counter() - started_at) * 1000,
            2
        )

        vector = embedding.tolist()

        return {
            "embedding": vector,
            "dimension": len(vector),
            "model": MODEL_NAME,
            "input_type": request.input_type,
            "normalized": True,
            "latency_ms": latency_ms
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Embedding generation failed: {str(exc)}"
        )