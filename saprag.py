#!/usr/bin/env python3
"""SAPRAG - Central TUI to run the document/RAG pipeline modules.

A single entry point that dispatches to the standalone scripts:

  - Insert document (interactive)   -> cutter.main()
  - Process new files (batch)       -> process_new_files.process()
  - Compute embeddings              -> embed_chunks.embed_chunks()
  - Search (RAG pipeline)           -> search.enhanced_search(query)
  - Evolve cached answers           -> evolution.main()
  - Migrate PostgreSQL -> Firestore -> migrate_to_firestore.migrate()

Each module is imported lazily so a missing optional dependency only affects the
action that needs it, not the whole menu.
"""

import sys

from simple_term_menu import TerminalMenu


def _pause():
    input("\nPress Enter to continue...")


def insert_document():
    """Interactive single-file chunking into PostgreSQL + Firestore."""
    import cutter
    cutter.main()


def process_new_files():
    """Batch-process files from arquivos_novos/ into Firestore."""
    import process_new_files
    process_new_files.process()
    _pause()


def compute_embeddings():
    """Compute embeddings for chunks that don't have one yet."""
    import embed_chunks
    embed_chunks.embed_chunks()
    _pause()


def run_search():
    """Run the full multi-query RAG pipeline for a typed question."""
    import search
    print()
    query = input("Enter your question: ").strip()
    if not query:
        print("Empty question.")
        _pause()
        return
    try:
        answer = search.enhanced_search(query)
    except RuntimeError as e:
        print(f"\n[Error] {e}")
        _pause()
        return
    print(f"\n{'='*70}")
    print("FINAL ANSWER:")
    print(f"{'='*70}\n")
    print(answer)
    _pause()


def run_evolution():
    """Improve the most recent cached answers by reading documents whole."""
    import evolution
    evolution.main()
    _pause()


def run_migration():
    """Migrate documents and chunks from PostgreSQL to Firestore."""
    import migrate_to_firestore
    migrate_to_firestore.migrate()
    _pause()


ACTIONS = [
    ("Insert document (interactive chunking)", insert_document),
    ("Process new files (arquivos_novos/)", process_new_files),
    ("Compute embeddings", compute_embeddings),
    ("Search (RAG pipeline)", run_search),
    ("Evolve cached answers", run_evolution),
    ("Migrate PostgreSQL -> Firestore", run_migration),
    ("Exit", None),
]


def main():
    while True:
        labels = [label for label, _ in ACTIONS]
        menu = TerminalMenu(labels, title="\n=== SAPRAG ===\n")
        choice = menu.show()

        if choice is None or ACTIONS[choice][1] is None:
            print("Goodbye!")
            break

        label, action = ACTIONS[choice]
        try:
            action()
        except KeyboardInterrupt:
            print("\nInterrupted. Back to menu.")
        except Exception as e:
            print(f"\n[Error] {label} failed: {e}")
            _pause()


if __name__ == "__main__":
    main()
