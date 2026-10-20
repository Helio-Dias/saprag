#!/usr/bin/env python3
"""Migrate documents and chunks from PostgreSQL to Firestore."""

import psycopg2
from google.cloud import firestore

DB_PARAMS = {
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "postgres",
    "dbname": "postgres",
}

GCP_PROJECT = "ems-data-platform-dev"
FIRESTORE_DATABASE = "helio-test-db"


def migrate():
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    db = firestore.Client(project=GCP_PROJECT, database=FIRESTORE_DATABASE)

    cur.execute("SELECT id, descricao, caminho_arquivo FROM cortador.documentos ORDER BY id")
    documents = cur.fetchall()

    if not documents:
        print("No documents found in PostgreSQL.")
        return

    print(f"Migrating {len(documents)} document(s)...")

    for doc_id, description, file_path in documents:
        cur.execute(
            "SELECT sequencia_chunk, texto FROM cortador.chunks WHERE id_documento = %s ORDER BY sequencia_chunk",
            (doc_id,),
        )
        chunks = cur.fetchall()

        doc_ref = db.collection("documentos").document(str(doc_id))
        doc_ref.set({
            "id": doc_id,
            "descricao": description,
            "caminho_arquivo": file_path,
            "total_chunks": len(chunks),
        })

        batch = db.batch()
        for seq, text in chunks:
            chunk_ref = doc_ref.collection("chunks").document(str(seq))
            batch.set(chunk_ref, {
                "id_documento": doc_id,
                "sequencia_chunk": seq,
                "texto": text,
                "embedding": None,
            })
        batch.commit()

        print(f"  Document {doc_id} ({description}): {len(chunks)} chunk(s) migrated")

    cur.close()
    conn.close()
    print("Migration complete.")


if __name__ == "__main__":
    migrate()
