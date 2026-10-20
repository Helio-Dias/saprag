#!/usr/bin/env python3
"""Evolution - Improve cached answers beyond what the RAG pipeline produced.

For the N most recent cached questions, read the stored documents directly
(no vector search / no RAG) and let the local Qwen model try to build a
superior answer. Each question gets an equal slice of the total minute budget.
Improved answers are written to the Firestore 'evolution' collection.

The premise: the RAG answer may have lost context by retrieving only a few
generic chunks. By letting the model read whole documents (via a map-reduce
over the full corpus) it can produce a more complete, grounded answer.

Reading every document whole with the LLM is slow, so a single run
rarely finishes a question within its time slice. Progress (next document to
read + evidence gathered so far) is persisted per question in the
'evolution_progress' collection. Re-running resumes where the previous run
stopped instead of starting over; the final answer is produced only once all
documents have been read.
"""

import datetime
import time

from google.cloud import firestore

from search import call_claude

GCP_PROJECT = "ems-data-platform-dev"
FIRESTORE_DATABASE = "helio-test-db"

# Per-document map step: how many chunks to join into one document text before
# summarizing it against the question. Keeps each map prompt well under the
# model's context window.
MAP_CHUNK_BATCH = 40


def get_firestore() -> firestore.Client:
    return firestore.Client(project=GCP_PROJECT, database=FIRESTORE_DATABASE)


def ask_int(prompt: str, minimum: int = 1) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
            if value < minimum:
                print(f"Enter an integer >= {minimum}.")
                continue
            return value
        except ValueError:
            print("Invalid value. Enter an integer.")


def load_recent_cache(db: firestore.Client, n: int) -> list[dict]:
    """Return the N most recently cached questions (newest first)."""
    entries = []
    for snap in db.collection("cache_resultado").stream():
        data = snap.to_dict()
        if not data.get("query_text"):
            continue
        data["_doc_id"] = snap.id
        entries.append(data)

    def cache_time(entry: dict):
        return entry.get("data_cache") or datetime.datetime.min.replace(
            tzinfo=datetime.timezone.utc
        )

    entries.sort(key=cache_time, reverse=True)
    return entries[:n]


def load_documents(db: firestore.Client) -> list[dict]:
    """Load every stored document with its chunks joined into full text.

    No embeddings or vector search are used — this reads the raw corpus so the
    model can reason over whole documents instead of a handful of chunks.
    Returned in a stable order (by document id) so the resume index stays valid
    across runs.
    """
    documents = []
    for doc_snap in db.collection("documentos").stream():
        doc_data = doc_snap.to_dict()

        data_onboard = doc_data.get("data_onboard")
        if data_onboard and hasattr(data_onboard, "strftime"):
            date_str = data_onboard.strftime("%Y-%m-%d")
        elif data_onboard:
            date_str = str(data_onboard)
        else:
            date_str = "unknown"

        chunk_snaps = list(
            db.collection("documentos")
            .document(doc_snap.id)
            .collection("chunks")
            .order_by("sequencia_chunk")
            .stream()
        )
        texts = [c.to_dict().get("texto", "") for c in chunk_snaps]

        documents.append({
            "id": doc_data.get("id"),
            "descricao": doc_data.get("descricao") or "(no description)",
            "data_onboard": date_str,
            "chunks": texts,
        })

    documents.sort(key=lambda d: (d["id"] is None, d["id"]))
    return documents


def load_progress(db: firestore.Client, cache_doc_id: str) -> dict:
    """Return saved progress for a question, or a fresh state if none exists."""
    snap = db.collection("evolution_progress").document(cache_doc_id).get()
    if snap.exists:
        data = snap.to_dict()
        return {
            "next_index": data.get("next_index", 0),
            "evidence_parts": data.get("evidence_parts", []),
        }
    return {"next_index": 0, "evidence_parts": []}


def save_progress(db: firestore.Client, cache_doc_id: str, next_index: int, evidence_parts: list[str]) -> None:
    db.collection("evolution_progress").document(cache_doc_id).set({
        "next_index": next_index,
        "evidence_parts": evidence_parts,
        "data_atualizacao": datetime.datetime.now(datetime.timezone.utc),
    })


def clear_progress(db: firestore.Client, cache_doc_id: str) -> None:
    db.collection("evolution_progress").document(cache_doc_id).delete()


def map_document(question: str, document: dict) -> str:
    """Map step: extract from one document only what is relevant to the question.

    Chunks are batched so a very large document is processed in several passes,
    then the partial extractions are concatenated.
    """
    chunks = document["chunks"]
    partials = []

    for i in range(0, len(chunks), MAP_CHUNK_BATCH):
        batch_text = "\n".join(chunks[i:i + MAP_CHUNK_BATCH])
        prompt = f"""You are extracting information from a technical document to help answer a question.

IMPORTANT: Respond in the SAME LANGUAGE as the QUESTION.

QUESTION:
{question}

DOCUMENT: {document['descricao']} (onboard date: {document['data_onboard']})

DOCUMENT TEXT (segment {i // MAP_CHUNK_BATCH + 1}):
{batch_text}

INSTRUCTIONS:
1. Extract ONLY the facts, steps, values, and passages from this text that are relevant to the QUESTION.
2. Quote or paraphrase faithfully. Do NOT invent anything that is not in the text.
3. If this segment contains nothing relevant to the question, respond with exactly: NONE
4. Be concise but keep every relevant detail (commands, numbers, configuration)."""
        extract = call_claude(prompt).strip()
        if extract and extract.upper() != "NONE":
            partials.append(extract)

    return "\n".join(partials)


def reduce_answer(question: str, cached_answer: str, evidence: str) -> str:
    """Reduce step: build a final answer from the extracted evidence, aiming to
    surpass the cached (RAG) answer."""
    prompt = f"""You are a technical assistant. You will produce the best possible answer to a question using evidence extracted directly from the full source documents (not a generic RAG retrieval).

IMPORTANT: Respond in the SAME LANGUAGE as the QUESTION.

QUESTION:
{question}

PREVIOUS ANSWER (produced by a RAG pipeline that may have lost context by using only a few generic chunks):
{cached_answer}

EVIDENCE EXTRACTED FROM THE FULL DOCUMENTS:
{evidence}

INSTRUCTIONS:
1. Write an answer that is MORE COMPLETE, MORE PRECISE, and better grounded than the PREVIOUS ANSWER.
2. Base yourself EXCLUSIVELY on the EVIDENCE above. Do NOT invent anything.
3. Fill gaps the previous answer left: truncated commands, missing steps, vague statements.
4. If evidence contradicts the previous answer, prefer the evidence and note the correction.
5. Indicate the source document for important claims when the evidence identifies it.
6. If the evidence genuinely does not improve on the previous answer, say so and return the best grounded answer you can."""
    return call_claude(prompt).strip()


def evolve_question(
    db: firestore.Client,
    entry: dict,
    documents: list[dict],
    deadline: float,
) -> None:
    """Advance one cached question's map-reduce, respecting a soft deadline.

    Resumes from persisted progress. The deadline is checked before reading each
    new document; a map call already running is always allowed to finish. When
    all documents have been read, runs the reduce step, writes to 'evolution',
    and clears the progress record.
    """
    cache_doc_id = entry["_doc_id"]
    question = entry["query_text"]
    cached_answer = entry.get("resposta", "")

    progress = load_progress(db, cache_doc_id)
    idx = progress["next_index"]
    evidence_parts = progress["evidence_parts"]

    if idx > 0:
        print(f"    Resuming from document {idx + 1}/{len(documents)} "
              f"({len(evidence_parts)} evidence block(s) so far).")

    while idx < len(documents):
        if time.monotonic() >= deadline:
            save_progress(db, cache_doc_id, idx, evidence_parts)
            print(f"    Time budget reached — stopped at document {idx}/{len(documents)}. Progress saved.")
            return

        doc = documents[idx]
        if doc["chunks"]:
            print(f"    Reading document {doc['id']}: {doc['descricao'][:60]}")
            extract = map_document(question, doc)
            if extract:
                evidence_parts.append(
                    f"[Document {doc['id']} | {doc['descricao']} | Onboard: {doc['data_onboard']}]\n{extract}"
                )
        idx += 1

    # All documents read — produce the final evolved answer.
    if not evidence_parts:
        print("    No relevant evidence found in any document. Keeping cached answer.")
        evolved_answer = cached_answer
    else:
        print(f"    All {len(documents)} documents read. Synthesizing improved answer...")
        evolved_answer = reduce_answer(question, cached_answer, "\n\n---\n\n".join(evidence_parts))

    save_evolution(db, cache_doc_id, question, cached_answer, evolved_answer)
    clear_progress(db, cache_doc_id)
    print("    Evolution complete and saved.")


def save_evolution(
    db: firestore.Client,
    cache_doc_id: str,
    question: str,
    previous_answer: str,
    evolved_answer: str,
) -> None:
    db.collection("evolution").add({
        "cache_doc_id": cache_doc_id,
        "query_text": question,
        "resposta_anterior": previous_answer,
        "resposta_evoluida": evolved_answer,
        "data_evolucao": datetime.datetime.now(datetime.timezone.utc),
    })


def main():
    print("=== EVOLUTION ===\n")
    total_minutes = ask_int("Minute budget (total): ")
    cache_count = ask_int("Number of cached questions to evolve: ")

    per_question_seconds = (total_minutes * 60) / cache_count
    print(
        f"\nBudget: {total_minutes} min over {cache_count} question(s) "
        f"= {per_question_seconds:.0f}s each.\n"
    )

    db = get_firestore()

    print("Loading recent cache entries...")
    cache_entries = load_recent_cache(db, cache_count)
    if not cache_entries:
        print("No cached questions found. Nothing to evolve.")
        return
    if len(cache_entries) < cache_count:
        print(f"Only {len(cache_entries)} cached question(s) available; evolving those.")

    print("Loading stored documents (full text, no RAG)...")
    documents = load_documents(db)
    print(f"  {len(documents)} document(s) loaded.")
    if not documents:
        print("No documents stored. Nothing to read.")
        return

    for i, entry in enumerate(cache_entries, 1):
        print(f"[{i}/{len(cache_entries)}] {entry['query_text'][:80]}")
        deadline = time.monotonic() + per_question_seconds
        try:
            evolve_question(db, entry, documents, deadline)
        except RuntimeError as e:
            print(f"    ERROR: {e}")
        print()

    print("Run finished. Re-run to continue any questions that are not yet complete.")


if __name__ == "__main__":
    main()
