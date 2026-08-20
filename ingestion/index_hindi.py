import polars as pl
import requests
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


# ============================================================
# Configuration
# ============================================================

PARQUET = "hintrain.parquet"

EMBED_URL = "http://127.0.0.1:8000/embed"

QDRANT_PATH = "C:/RAG-Goa/qdrant"
COLLECTION = "msmarco_hindi"

BATCH_SIZE = 16
MAX_RECORDS = 100


# ============================================================
# Connect to Qdrant
# ============================================================

client = QdrantClient(
    path=QDRANT_PATH
)


# ============================================================
# Load Hindi dataset
# ============================================================

df = pl.read_parquet(
    PARQUET,
    columns=["query", "passages"],
    n_rows=MAX_RECORDS
)


# ============================================================
# Index selected passages
# ============================================================

points = []
indexed = 0
skipped = 0


for row in df.iter_rows(named=True):

    query = row["query"]
    passages = row["passages"]

    translated_passages = passages["Translated_passages"]
    selected_passages = passages["is_selected"]


    # --------------------------------------------------------
    # Process passages
    # --------------------------------------------------------

    for i, text in enumerate(translated_passages):

        # Only index passages marked as relevant
        if not selected_passages[i]:
            skipped += 1
            continue

        # Ignore empty / extremely short passages
        if not text or len(text.strip()) < 30:
            skipped += 1
            continue

        text = text.strip()


        # ====================================================
        # Generate passage embedding
        #
        # E5 requires:
        # passage: <text>
        # ====================================================

        response = requests.post(
            EMBED_URL,
            json={
                "text": text,
                "input_type": "passage"
            },
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        embedding = data["embedding"]


        # ----------------------------------------------------
        # Validate embedding
        # ----------------------------------------------------

        if len(embedding) != 384:
            raise ValueError(
                f"Expected 384-dimensional embedding, "
                f"got {len(embedding)}"
            )


        # ====================================================
        # Qdrant payload
        # ====================================================

        payload = {
            "text": text,
            "query": query,
            "source": "MSMARCO-XI",
            "language": "hi",
            "passage_index": i,
            "is_selected": True,
            "chunk_strategy": "passage"
        }


        # ====================================================
        # Create Qdrant point
        # ====================================================

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload
            )
        )


        # ====================================================
        # Batch upload
        # ====================================================

        if len(points) >= BATCH_SIZE:

            client.upsert(
                collection_name=COLLECTION,
                points=points
            )

            indexed += len(points)

            print(f"Indexed: {indexed}")

            points = []


# ============================================================
# Upload remaining points
# ============================================================

if points:

    client.upsert(
        collection_name=COLLECTION,
        points=points
    )

    indexed += len(points)


# ============================================================
# Final statistics
# ============================================================

print()
print("=" * 60)
print("INDEXING COMPLETE")
print("=" * 60)

print(f"Records processed : {MAX_RECORDS}")
print(f"Vectors indexed   : {indexed}")
print(f"Passages skipped  : {skipped}")
print(f"Collection        : {COLLECTION}")
print(f"Embedding size    : 384")

print("=" * 60)


# ============================================================
# Close Qdrant
# ============================================================

client.close()