#!/usr/bin/env python3
"""Compute embeddings for chunks in Firestore using Vertex AI text-embedding-004."""

import vertexai
from vertexai.language_models import TextEmbeddingModel
from google.cloud import firestore

GCP_PROJECT = "ems-data-platform-dev"
GCP_LOCATION = "europe-west1"
FIRESTORE_DATABASE = "helio-test-db"
EMBEDDING_MODEL = "text-embedding-004"
BATCH_SIZE = 20  # API token limit (~20k tokens per call)


def embed_chunks():
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    db = firestore.Client(project=GCP_PROJECT, database=FIRESTORE_DATABASE)

    documents = list(db.collection("documentos").stream())
    if not documents:
        print("No documents found in Firestore.")
        return

    print(f"Processing {len(documents)} document(s)...")
    total_embedded = 0
    total_skipped = 0

    for doc_snap in documents:
        doc_id = doc_snap.id
        doc_data = doc_snap.to_dict()

        chunks = list(
            db.collection("documentos").document(doc_id).collection("chunks")
            .order_by("sequencia_chunk")
            .stream()
        )

        # keep only chunks without a computed embedding
        pending = [c for c in chunks if c.to_dict().get("embedding") is None]
        total_skipped += len(chunks) - len(pending)

        if not pending:
            print(f"  Document {doc_id} ({doc_data.get('descricao')}): all chunks already have an embedding.")
            continue

        print(f"  Document {doc_id} ({doc_data.get('descricao')}): {len(pending)} chunk(s) to embed...")

        # process in batches
        for i in range(0, len(pending), BATCH_SIZE):
            batch_items = pending[i:i + BATCH_SIZE]
            texts = [c.to_dict()["texto"] for c in batch_items]

            results = model.get_embeddings(texts)

            batch = db.batch()
            for chunk_snap, result in zip(batch_items, results):
                batch.update(chunk_snap.reference, {"embedding": result.values})
            batch.commit()

            total_embedded += len(batch_items)
            print(f"    Batch {i // BATCH_SIZE + 1}: {len(batch_items)} embedding(s) written.")

    print(f"\nDone. {total_embedded} embedding(s) computed, {total_skipped} already existed.")


if __name__ == "__main__":
    embed_chunks()
