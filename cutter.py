#!/usr/bin/env python3
"""Cutter - Split documents into chunks and store them in PostgreSQL and Firestore."""

import os
import sys

from simple_term_menu import TerminalMenu
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

_firestore_client = None


def get_firestore():
    global _firestore_client
    if _firestore_client is None:
        _firestore_client = firestore.Client(project=GCP_PROJECT, database=FIRESTORE_DATABASE)
    return _firestore_client


def next_firestore_id(db) -> int:
    """Next document id based on the Firestore store (max existing id + 1)."""
    docs = list(db.collection("documentos").stream())
    if not docs:
        return 1
    return max(int(d.id) for d in docs) + 1


def read_file(path: str) -> str:
    """Read text from .txt or .doc (plain text) files."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()

state = {
    "file": None,
    "description": None,
    "chunk_size": 528,
    "overlap": 128,
}


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks with overlap, respecting word boundaries."""
    if not text.strip():
        return []

    chunks = []
    pos = 0
    length = len(text)

    while pos < length:
        end = pos + chunk_size

        if end >= length:
            segment = text[pos:].strip()
            if segment:
                chunks.append(segment)
            break

        while end < length and not text[end].isspace():
            end += 1

        segment = text[pos:end].strip()
        if segment:
            chunks.append(segment)

        next_pos = end - overlap

        if next_pos < length and next_pos > 0 and not text[next_pos].isspace():
            while next_pos < length and not text[next_pos].isspace():
                next_pos += 1

        while next_pos < length and text[next_pos].isspace():
            next_pos += 1

        if next_pos <= pos:
            next_pos = end
            while next_pos < length and text[next_pos].isspace():
                next_pos += 1

        pos = next_pos

    return chunks


def get_connection():
    return psycopg2.connect(**DB_PARAMS)


def choose_file():
    files = sorted(
        f for f in os.listdir(".")
        if os.path.isfile(f) and not f.startswith(".")
        and f.lower().endswith((".txt", ".doc", ".docx", ".md"))
    )
    if not files:
        print("\nNo files found in the current directory.")
        input("Press Enter to continue...")
        return

    menu = TerminalMenu(files, title="\nChoose a file:\n")
    idx = menu.show()
    if idx is not None:
        state["file"] = os.path.abspath(files[idx])


def choose_description():
    print()
    desc = input("Enter the document description: ").strip()
    if desc:
        state["description"] = desc
    else:
        print("Description cannot be empty.")
        input("Press Enter to continue...")


def choose_chunk_size():
    print()
    value = input(f"Enter the chunk size (current: {state['chunk_size']}): ").strip()
    if not value:
        return
    try:
        n = int(value)
        if n <= 0:
            raise ValueError
        state["chunk_size"] = n
    except ValueError:
        print("Invalid value. Enter a positive integer.")
        input("Press Enter to continue...")


def choose_overlap():
    print()
    value = input(f"Enter the overlap (current: {state['overlap']}): ").strip()
    if not value:
        return
    try:
        n = int(value)
        if n < 0:
            raise ValueError
        state["overlap"] = n
    except ValueError:
        print("Invalid value. Enter a non-negative integer.")
        input("Press Enter to continue...")


def run_chunking():
    print()

    if not state["file"]:
        print("Error: no file selected.")
        input("Press Enter to continue...")
        return

    if not state["description"]:
        print("Error: description not provided.")
        input("Press Enter to continue...")
        return

    if state["overlap"] >= state["chunk_size"]:
        print("Error: overlap must be smaller than chunk size.")
        input("Press Enter to continue...")
        return

    try:
        text = read_file(state["file"])
    except FileNotFoundError:
        print(f"Error: file not found: {state['file']}")
        input("Press Enter to continue...")
        return
    except UnicodeDecodeError:
        print("Error: the file is not encoded in UTF-8 or latin-1.")
        input("Press Enter to continue...")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        input("Press Enter to continue...")
        return

    chunks = chunk_text(text, state["chunk_size"], state["overlap"])

    if not chunks:
        print("Empty file or no text content.")
        input("Press Enter to continue...")
        return

    oversized = [i for i, c in enumerate(chunks, 1) if len(c) > 1000]
    if oversized:
        print(f"Warning: {len(oversized)} chunk(s) exceed 1000 characters (DB limit).")
        print("Reduce the chunk size or check the file.")
        input("Press Enter to continue...")
        return

    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            "INSERT INTO cortador.documentos (descricao, caminho_arquivo) VALUES (%s, %s) RETURNING id",
            (state["description"], state["file"]),
        )
        doc_id = cur.fetchone()[0]

        for seq, chunk in enumerate(chunks, 1):
            cur.execute(
                "INSERT INTO cortador.chunks (id_documento, sequencia_chunk, texto) VALUES (%s, %s, %s)",
                (doc_id, seq, chunk),
            )

        conn.commit()
        print(f"\nSuccess! Document ID: {doc_id}")
        print(f"Total chunks inserted: {len(chunks)}")

        try:
            db = get_firestore()
            # Firestore ids are allocated independently from Postgres SERIAL ids.
            # Deriving the id from Firestore (not reusing doc_id) keeps this write
            # consistent with process_new_files.py and prevents overwriting an
            # existing document when the two stores are out of sync.
            fs_id = next_firestore_id(db)
            doc_ref = db.collection("documentos").document(str(fs_id))
            # create() fails if the document already exists, so a stale id can
            # never silently overwrite real data.
            doc_ref.create({
                "id": fs_id,
                "descricao": state["description"],
                "caminho_arquivo": state["file"],
                "total_chunks": len(chunks),
            })
            batch = db.batch()
            for seq, chunk in enumerate(chunks, 1):
                chunk_ref = doc_ref.collection("chunks").document(str(seq))
                batch.set(chunk_ref, {
                    "id_documento": fs_id,
                    "sequencia_chunk": seq,
                    "texto": chunk,
                    "embedding": None,
                })
            batch.commit()
            print(f"Firestore: document {fs_id} and {len(chunks)} chunk(s) written.")
        except Exception as e:
            print(f"Warning: error writing to Firestore: {e}")

    except psycopg2.OperationalError:
        print("Error: could not connect to the database.")
        print("Check that Docker/Colima is running.")
    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

    input("\nPress Enter to continue...")


def main():
    while True:
        file_display = os.path.basename(state["file"]) if state["file"] else "none"
        desc_display = state["description"] or "none"

        options = [
            f"Choose file          [{file_display}]",
            f"Choose description   [{desc_display}]",
            f"Choose chunk size    [{state['chunk_size']}]",
            f"Choose overlap       [{state['overlap']}]",
            "Run chunking",
            "Exit",
        ]

        menu = TerminalMenu(options, title="\n=== CUTTER ===\n")
        choice = menu.show()

        if choice is None or choice == 5:
            print("Goodbye!")
            break
        elif choice == 0:
            choose_file()
        elif choice == 1:
            choose_description()
        elif choice == 2:
            choose_chunk_size()
        elif choice == 3:
            choose_overlap()
        elif choice == 4:
            run_chunking()


if __name__ == "__main__":
    main()
