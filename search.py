#!/usr/bin/env python3
"""Multi-query semantic search with two-round RAG refinement."""

import datetime
import json
import math
import subprocess
import sys

import vertexai
from vertexai.language_models import TextEmbeddingModel
from google.cloud import firestore

GCP_PROJECT = "ems-data-platform-dev"
GCP_LOCATION = "europe-west1"
FIRESTORE_DATABASE = "helio-test-db"
EMBEDDING_MODEL = "text-embedding-004"

# LLM used for query decomposition and RAG synthesis: Claude Haiku, reached
# through the Claude Code CLI (`claude -p`).
CLAUDE_CLI = "claude"
CLAUDE_MODEL = "claude-haiku-4-5"

TOP_K_PER_QUERY = 5
TOP_K_MERGED = 15
TOP_K_SECOND_ROUND = 10
MAX_SUB_QUERIES = 4

# Cache thresholds (cosine similarity)
CACHE_PERFECT_THRESHOLD = 0.98
CACHE_NEAR_THRESHOLD = 0.92
CACHE_AUX_THRESHOLD = 0.85


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# --- Cache functions ---

def load_cache(db: firestore.Client) -> list[dict]:
    entries = []
    for snap in db.collection("cache_resultado").stream():
        data = snap.to_dict()
        if not data.get("query_embedding"):
            continue
        data["_doc_id"] = snap.id
        entries.append(data)
    return entries


def check_cache(query_embedding: list[float], cache_entries: list[dict]) -> tuple[str, dict | None]:
    best_score = 0.0
    best_entry = None

    for entry in cache_entries:
        score = cosine_similarity(query_embedding, entry["query_embedding"])
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry is None:
        return ("miss", None)

    if best_score >= CACHE_PERFECT_THRESHOLD:
        return ("perfect", best_entry)
    elif best_score >= CACHE_NEAR_THRESHOLD:
        return ("near", best_entry)
    elif best_score >= CACHE_AUX_THRESHOLD:
        return ("aux", best_entry)
    else:
        return ("miss", None)


def save_cache(db: firestore.Client, query_text: str, query_embedding: list[float], resposta: str):
    now = datetime.datetime.now(datetime.timezone.utc)
    db.collection("cache_resultado").add({
        "query_text": query_text,
        "query_embedding": query_embedding,
        "resposta": resposta,
        "data_cache": now,
        "data_ultimo_uso": now,
        "uso_exclusivo": 0,
        "uso_auxiliar": 0,
    })


def update_cache_stats(db: firestore.Client, cache_doc_id: str, hit_type: str):
    now = datetime.datetime.now(datetime.timezone.utc)
    ref = db.collection("cache_resultado").document(cache_doc_id)
    if hit_type in ("perfect", "near"):
        ref.update({
            "data_ultimo_uso": now,
            "uso_exclusivo": firestore.Increment(1),
        })
    else:
        ref.update({
            "data_ultimo_uso": now,
            "uso_auxiliar": firestore.Increment(1),
        })


# --- End cache functions ---


def call_claude(prompt: str, timeout: int = 120) -> str:
    """Run a prompt through Claude Haiku via the Claude Code CLI (`claude -p`)."""
    try:
        result = subprocess.run(
            [CLAUDE_CLI, "-p", "--model", CLAUDE_MODEL, "--output-format", "text"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"Claude CLI not found ('{CLAUDE_CLI}'). Install Claude Code and ensure it is on PATH."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Claude CLI timed out after {timeout}s.")

    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI error (exit {result.returncode}): {result.stderr.strip()}")

    return result.stdout.strip()


def generate_sub_queries(original_query: str) -> list[str]:
    prompt = f"""You are an expert in search query decomposition. Your task is to break down the user's question into specific sub-queries to retrieve information from a technical knowledge base.

IMPORTANT: Generate the sub-queries in the SAME LANGUAGE as the original query below.

ORIGINAL QUERY:
{original_query}

INSTRUCTIONS:
1. Generate 2 to 4 sub-queries that together cover ALL aspects of the original question.
2. CRITICAL: Each sub-query MUST maintain the central context of the original query.
   - If the original asks about "MySQL temporary tablespace issues on GCP",
     every sub-query must reference MySQL/GCP/temporary tablespace.
   - Do NOT generate generic sub-queries like "how to monitor disk" without the specific context.
3. Sub-queries should target different angles:
   - Root cause / diagnosis
   - Solution / remediation steps
   - Monitoring / prevention
   - Related configuration or architecture
4. Each sub-query must be a complete, self-contained search phrase.
5. The sub-queries MUST be in the same language as the ORIGINAL QUERY.

OUTPUT FORMAT:
Return ONLY a JSON array of strings, nothing else. Example:
["sub-query 1", "sub-query 2", "sub-query 3"]"""

    try:
        response = call_claude(prompt)
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            queries = json.loads(response[start:end])
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return queries[:MAX_SUB_QUERIES]
    except (RuntimeError, json.JSONDecodeError, ValueError):
        pass
    return []


def load_all_chunks(db: firestore.Client) -> list[dict]:
    chunks = []
    for doc_snap in db.collection("documentos").stream():
        doc_data = doc_snap.to_dict()
        data_onboard = doc_data.get("data_onboard")
        if data_onboard:
            date_str = data_onboard.strftime("%Y-%m-%d") if hasattr(data_onboard, "strftime") else str(data_onboard)
        else:
            date_str = "unknown"

        for chunk_snap in db.collection("documentos").document(doc_snap.id).collection("chunks").stream():
            chunk = chunk_snap.to_dict()
            if not chunk.get("embedding"):
                continue
            chunks.append({
                "doc_id": doc_data.get("id"),
                "descricao": doc_data.get("descricao"),
                "data_onboard": date_str,
                "sequencia": chunk["sequencia_chunk"],
                "texto": chunk["texto"],
                "embedding": chunk["embedding"],
            })
    return chunks


def embed_queries(model, queries: list[str]) -> list[list[float]]:
    results = model.get_embeddings(queries)
    return [r.values for r in results]


def multi_query_search(
    query_embeddings: list[list[float]],
    chunks: list[dict],
    top_k_per_query: int = TOP_K_PER_QUERY,
    top_k_merged: int = TOP_K_MERGED,
) -> list[dict]:
    best_scores: dict[tuple, dict] = {}

    for q_emb in query_embeddings:
        scored = []
        for chunk in chunks:
            score = cosine_similarity(q_emb, chunk["embedding"])
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)

        for score, chunk in scored[:top_k_per_query]:
            key = (chunk["doc_id"], chunk["sequencia"])
            if key not in best_scores or score > best_scores[key]["score"]:
                best_scores[key] = {
                    "score": score,
                    "doc_id": chunk["doc_id"],
                    "descricao": chunk["descricao"],
                    "data_onboard": chunk["data_onboard"],
                    "sequencia": chunk["sequencia"],
                    "texto": chunk["texto"],
                }

    merged = sorted(best_scores.values(), key=lambda x: x["score"], reverse=True)
    return merged[:top_k_merged]


def format_chunks_for_prompt(resultados: list[dict]) -> str:
    lines = []
    for i, r in enumerate(resultados, 1):
        lines.append(
            f"[Chunk {i} | Doc: {r['descricao']} | Onboard: {r['data_onboard']} | Score: {r['score']:.4f}]\n{r['texto']}"
        )
    return "\n\n---\n\n".join(lines)


def synthesize_with_followups(original_query: str, resultados: list[dict]) -> tuple[str, list[str]]:
    chunks_txt = format_chunks_for_prompt(resultados)

    prompt = f"""You are a technical assistant. Below are text chunks retrieved from a knowledge base.

IMPORTANT: Your entire response MUST be in the SAME LANGUAGE as the user's question below. If the question is in English, respond in English. If in Portuguese, respond in Portuguese. Match the language exactly.

USER QUESTION:
{original_query}

RETRIEVED CHUNKS (ordered by relevance, most to least relevant):

{chunks_txt}

INSTRUCTIONS:
1. Answer the user's question based EXCLUSIVELY on the information in the chunks above.
2. Do NOT invent, assume, or add anything not explicitly in the chunks.
3. If there are contradictions between chunks, point them out clearly.
4. To resolve contradictions, use the most recent chunk (latest onboard date) as source of truth, or if dates are equal, follow the majority of chunks.
5. Always indicate the source of each important claim (document name and onboard date).
6. If the chunks do not contain enough information to answer, say so clearly.

After your response, add the following section in EXACTLY this format:

<<<FOLLOW_UP>>>
[If the response above is complete and conclusive, write only: NONE]
[Otherwise, list follow-up questions as a JSON array of strings. Each question should seek missing or incomplete information.]
<<<END_FOLLOW_UP>>>

Generate follow-up questions ONLY if:
- Some information came truncated or partial (e.g., a SQL query cut in the middle)
- The answer is inconclusive (e.g., "it's done in a good way" without detailing how)
- There are clear gaps that additional chunks could fill

Each follow-up question MUST maintain the original context of "{original_query}" to stay in scope.
Each follow-up question MUST be in the same language as the original query."""

    response = call_claude(prompt, timeout=180)
    return _parse_followups(response)


def synthesize_with_followups_and_aux(
    original_query: str, resultados: list[dict], aux_response: str
) -> tuple[str, list[str]]:
    chunks_txt = format_chunks_for_prompt(resultados)

    prompt = f"""You are a technical assistant. Below are text chunks retrieved from a knowledge base, plus a CACHED RESPONSE from a previous similar query that may contain relevant information.

IMPORTANT: Your entire response MUST be in the SAME LANGUAGE as the user's question below. If the question is in English, respond in English. If in Portuguese, respond in Portuguese. Match the language exactly.

USER QUESTION:
{original_query}

CACHED RESPONSE FROM SIMILAR QUERY (use as additional context, but prioritize the chunks below for accuracy):
{aux_response}

RETRIEVED CHUNKS (ordered by relevance, most to least relevant):

{chunks_txt}

INSTRUCTIONS:
1. Answer the user's question based on the chunks above AND the cached response as supplementary context.
2. Prioritize information from the chunks over the cached response when there are conflicts.
3. Do NOT invent, assume, or add anything not in the chunks or cached response.
4. If there are contradictions between chunks, point them out clearly.
5. To resolve contradictions, use the most recent chunk (latest onboard date) as source of truth, or if dates are equal, follow the majority of chunks.
6. Always indicate the source of each important claim (document name and onboard date).
7. If the chunks do not contain enough information to answer, say so clearly.

After your response, add the following section in EXACTLY this format:

<<<FOLLOW_UP>>>
[If the response above is complete and conclusive, write only: NONE]
[Otherwise, list follow-up questions as a JSON array of strings. Each question should seek missing or incomplete information.]
<<<END_FOLLOW_UP>>>

Generate follow-up questions ONLY if:
- Some information came truncated or partial (e.g., a SQL query cut in the middle)
- The answer is inconclusive (e.g., "it's done in a good way" without detailing how)
- There are clear gaps that additional chunks could fill

Each follow-up question MUST maintain the original context of "{original_query}" to stay in scope.
Each follow-up question MUST be in the same language as the original query."""

    response = call_claude(prompt, timeout=180)
    return _parse_followups(response)


def _parse_followups(response: str) -> tuple[str, list[str]]:
    marker_start = "<<<FOLLOW_UP>>>"
    marker_end = "<<<END_FOLLOW_UP>>>"

    if marker_start not in response:
        return response.strip(), []

    parts = response.split(marker_start)
    answer = parts[0].strip()
    followup_section = parts[1].split(marker_end)[0].strip()

    if not followup_section or followup_section.upper() == "NONE":
        return answer, []

    try:
        start = followup_section.find("[")
        end = followup_section.rfind("]") + 1
        if start >= 0 and end > start:
            questions = json.loads(followup_section[start:end])
            if isinstance(questions, list):
                return answer, [q for q in questions if isinstance(q, str)][:MAX_SUB_QUERIES]
    except (json.JSONDecodeError, ValueError):
        pass

    lines = [l.strip().lstrip("- ").lstrip("0123456789.) ")
             for l in followup_section.splitlines() if l.strip()]
    questions = [l for l in lines if len(l) > 10 and l.upper() != "NONE"]
    return answer, questions[:MAX_SUB_QUERIES]


def synthesize_final(
    original_query: str,
    first_round_chunks: list[dict],
    second_round_chunks: list[dict],
    first_answer: str,
    follow_up_questions: list[str],
) -> str:
    first_chunks_txt = format_chunks_for_prompt(first_round_chunks)
    second_chunks_txt = format_chunks_for_prompt(second_round_chunks)
    followups_txt = "\n".join(f"- {q}" for q in follow_up_questions)

    prompt = f"""You are a technical assistant. The user asked a question and we already performed two rounds of search in the knowledge base.

IMPORTANT: Your entire response MUST be in the SAME LANGUAGE as the user's original question below. Match the language exactly.

USER'S ORIGINAL QUESTION:
{original_query}

FIRST ROUND RESPONSE (partial/incomplete):
{first_answer}

ASPECTS THAT REMAINED INCOMPLETE (motivated the second search):
{followups_txt}

FIRST ROUND CHUNKS:
{first_chunks_txt}

ADDITIONAL CHUNKS (SECOND ROUND - seeking complementary information):
{second_chunks_txt}

INSTRUCTIONS:
1. Produce a FINAL and COMPLETE answer to the user's original question.
2. Integrate information from BOTH search rounds.
3. Prioritize second-round chunks to fill the identified gaps.
4. Base yourself EXCLUSIVELY on chunk information. Do not invent anything.
5. If there are still unanswered aspects after two rounds, explicitly state what was not found in the knowledge base.
6. Indicate sources (document and date) for important claims.
7. For contradictions, use the most recent chunk as source of truth."""

    return call_claude(prompt, timeout=180)


def display_results(resultados: list[dict], label: str = "") -> None:
    header = f"Top {len(resultados)} chunks"
    if label:
        header += f" ({label})"
    print(f"\n{'='*70}")
    print(header)
    print(f"{'='*70}\n")
    for i, r in enumerate(resultados, 1):
        print(f"[{i}] Score: {r['score']:.4f} | Doc {r['doc_id']} | Chunk {r['sequencia']} | Onboard: {r['data_onboard']}")
        print(f"    {r['descricao']}")
        print(f"    {r['texto'][:200]}{'...' if len(r['texto']) > 200 else ''}")
        print()


def enhanced_search(query: str) -> str:
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    db = firestore.Client(project=GCP_PROJECT, database=FIRESTORE_DATABASE)

    # Step 0: Embed query and check cache
    print("Computing query embedding...")
    query_embedding = model.get_embeddings([query])[0].values

    print("Checking cache...")
    cache_entries = load_cache(db)
    cache_hit_type, cache_entry = check_cache(query_embedding, cache_entries)

    aux_response = None

    if cache_hit_type in ("perfect", "near"):
        update_cache_stats(db, cache_entry["_doc_id"], cache_hit_type)
        print(f"  Cache HIT ({cache_hit_type}) — cached query: {cache_entry.get('query_text', '?')[:80]}")
        print(f"  Returning cached response.")
        return cache_entry["resposta"]
    elif cache_hit_type == "aux":
        update_cache_stats(db, cache_entry["_doc_id"], cache_hit_type)
        aux_response = cache_entry["resposta"]
        print(f"  Auxiliary cache found — query: {cache_entry.get('query_text', '?')[:80]}")
        print(f"  Response will be used as additional context.")
    else:
        print(f"  Cache MISS. Running the full pipeline.")

    # Step 1: Sub-query generation
    print("\nGenerating sub-queries...")
    sub_queries = generate_sub_queries(query)
    all_queries = [query] + sub_queries
    print(f"  Original query: {query}")
    for i, sq in enumerate(sub_queries, 1):
        print(f"  Sub-query {i}: {sq}")

    # Step 2: Load all chunks once
    print("\nLoading chunks from Firestore...")
    chunks = load_all_chunks(db)
    print(f"  {len(chunks)} chunks with embedding loaded.")

    # Step 3: Batch embed all queries (reuse query_embedding for the original)
    print("\nComputing sub-query embeddings...")
    if sub_queries:
        sub_embeddings = embed_queries(model, sub_queries)
        query_embeddings = [query_embedding] + sub_embeddings
    else:
        query_embeddings = [query_embedding]

    # Step 4: Multi-query search (round 1)
    print("Searching (round 1)...")
    first_round_results = multi_query_search(
        query_embeddings, chunks,
        top_k_per_query=TOP_K_PER_QUERY,
        top_k_merged=TOP_K_MERGED,
    )
    display_results(first_round_results, label="Round 1")

    # Step 5: First synthesis with follow-up generation
    print("Synthesizing (round 1)...")
    if aux_response:
        first_answer, follow_up_questions = synthesize_with_followups_and_aux(
            query, first_round_results, aux_response
        )
    else:
        first_answer, follow_up_questions = synthesize_with_followups(query, first_round_results)

    print(f"\n{'='*70}")
    print("Response (round 1):")
    print(f"{'='*70}\n")
    print(first_answer)

    # Step 6: Check if second round needed
    if not follow_up_questions:
        print(f"\n{'='*70}")
        print("Complete response on the first round. No follow-ups needed.")
        print(f"{'='*70}")
        save_cache(db, query, query_embedding, first_answer)
        print("Cache saved.")
        return first_answer

    print(f"\n{'='*70}")
    print("Follow-up questions for the second round:")
    print(f"{'='*70}")
    for i, q in enumerate(follow_up_questions, 1):
        print(f"  {i}. {q}")

    # Step 7: Embed follow-up queries
    print("\nComputing follow-up embeddings...")
    followup_embeddings = embed_queries(model, follow_up_questions)

    # Step 8: Multi-query search (round 2) - reuse loaded chunks
    print("Searching (round 2)...")
    second_round_results = multi_query_search(
        followup_embeddings, chunks,
        top_k_per_query=TOP_K_PER_QUERY,
        top_k_merged=TOP_K_SECOND_ROUND,
    )

    # Remove chunks already seen in round 1
    first_round_keys = {(r["doc_id"], r["sequencia"]) for r in first_round_results}
    second_round_results = [
        r for r in second_round_results
        if (r["doc_id"], r["sequencia"]) not in first_round_keys
    ]

    if not second_round_results:
        print("\nSecond round brought no new chunks. Keeping the round 1 response.")
        save_cache(db, query, query_embedding, first_answer)
        print("Cache saved.")
        return first_answer

    display_results(second_round_results, label="Round 2 - new chunks")

    # Step 9: Final synthesis
    print("Synthesizing (final)...")
    try:
        final_answer = synthesize_final(
            query, first_round_results, second_round_results,
            first_answer, follow_up_questions,
        )
    except RuntimeError:
        print("[Warning] Final synthesis failed. Returning the round 1 response.")
        save_cache(db, query, query_embedding, first_answer)
        print("Cache saved.")
        return first_answer

    save_cache(db, query, query_embedding, final_answer)
    print("Cache saved.")
    return final_answer


# --- Legacy functions (backward compatibility) ---

def search(query: str, top_k: int = 10) -> list[dict]:
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    db = firestore.Client(project=GCP_PROJECT, database=FIRESTORE_DATABASE)

    query_embedding = model.get_embeddings([query])[0].values
    results = []

    for doc_snap in db.collection("documentos").stream():
        doc_data = doc_snap.to_dict()
        data_onboard = doc_data.get("data_onboard")
        if data_onboard:
            date_str = data_onboard.strftime("%Y-%m-%d") if hasattr(data_onboard, "strftime") else str(data_onboard)
        else:
            date_str = "unknown"

        for chunk_snap in db.collection("documentos").document(doc_snap.id).collection("chunks").stream():
            chunk = chunk_snap.to_dict()
            if not chunk.get("embedding"):
                continue
            score = cosine_similarity(query_embedding, chunk["embedding"])
            results.append({
                "score": score,
                "doc_id": doc_data.get("id"),
                "descricao": doc_data.get("descricao"),
                "data_onboard": date_str,
                "sequencia": chunk["sequencia_chunk"],
                "texto": chunk["texto"],
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def synthesize_with_claude(query: str, resultados: list[dict]) -> str:
    chunks_txt = format_chunks_for_prompt(resultados)

    prompt = f"""You are a technical assistant. Below are text chunks retrieved from a knowledge base.

IMPORTANT: Your entire response MUST be in the SAME LANGUAGE as the user's question below.

USER QUESTION:
{query}

RETRIEVED CHUNKS (ordered by relevance, most to least relevant):

{chunks_txt}

INSTRUCTIONS:
1. Answer the user's question based EXCLUSIVELY on the information in the chunks above.
2. Do NOT invent, assume, or add anything not explicitly in the chunks.
3. If there are contradictions between chunks, point them out clearly.
4. To resolve contradictions, use the most recent chunk (latest onboard date) as source of truth, or if dates are equal, follow the majority of chunks.
5. Always indicate the source of each important claim (document name and onboard date).
6. If the chunks do not contain enough information to answer, say so clearly."""

    return call_claude(prompt)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Query: ").strip()

    try:
        resposta = enhanced_search(query)
        print(f"\n{'='*70}")
        print("FINAL ANSWER:")
        print(f"{'='*70}\n")
        print(resposta)
    except RuntimeError as e:
        print(f"\n[Error] {e}", file=sys.stderr)
        sys.exit(1)
