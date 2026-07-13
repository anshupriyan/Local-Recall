import os
import sys
import numpy as np

# Ensure dependencies are available before loading heavier imports
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not found.")
    print("Please execute chat.py using the project virtual environment runner:")
    print("  .\\.venv\\Scripts\\python chat.py")
    sys.exit(1)

import config
from storage.db import search_snapshots
from recall_engine import get_embedding_model, build_context_block, query_local_llm

def print_welcome():
    print("=" * 80)
    print("LOCAL RECALL - Interactive Chat CLI")
    print("=" * 80)
    print("Ask questions about your captured screen history.")
    print("Type 'exit' or 'quit' to close, or press Ctrl+C.")
    print(f"Connecting to Local LLM: {config.LM_STUDIO_BASE_URL} (Model: {config.LM_STUDIO_MODEL_NAME})")
    print("=" * 80)

def main():
    db_path = config.DATABASE_PATH
    if not os.path.exists(db_path):
        print(f"ERROR: Database file '{db_path}' not found.")
        print("Please ensure the capture loop (main.py) has run and captured snapshots.")
        sys.exit(1)

    print_welcome()

    # Load embedding model once on startup via the shared engine
    try:
        model = get_embedding_model()
    except Exception as e:
        print(f"ERROR: Failed to load embedding model: {e}")
        sys.exit(1)

    while True:
        try:
            query = input("Ask Local Recall> ").strip()
            if not query:
                continue
                
            if query.lower() in ("exit", "quit"):
                print("Exiting Chat CLI.")
                break
                
            print("\nSearching screen history...")
            # 1. Embed query
            try:
                query_vector = model.encode(query)
                query_vec_bytes = query_vector.astype(np.float32).tobytes()
            except Exception as e:
                print(f"ERROR: Failed to generate query embedding: {e}\n")
                continue
                
            # 2. Retrieve context from DB
            try:
                results = search_snapshots(db_path, query_vec_bytes, top_k=5)
            except Exception as e:
                print(f"ERROR: Database search failed: {e}\n")
                continue
                
            if not results:
                print("No relevant snapshots found in your history.\n")
                continue
                
            # 3. Compile context block
            context_block = build_context_block(results)
            
            # 4. Request completions from local LLM
            print("Formulating answer using local LLM...")
            answer = query_local_llm(query, context_block)
            
            # 5. Output response
            print("\nAnswer:")
            print("-" * 80)
            print(answer)
            print("-" * 80)
            
            # 6. Output sources
            print("Sources:")
            for idx, row in enumerate(results):
                app = row['app_name'] or 'None'
                title = row['window_title'] or 'None'
                timestamp = row['timestamp']
                print(f"  [{idx+1}] Snapshot ID {row['id']} | Captured: {timestamp} | App: {app} | Title: {title}")
            print("=" * 80 + "\n")
            
        except (KeyboardInterrupt, EOFError):
            print("\nExiting Chat CLI.")
            break
        except Exception as e:
            print(f"\nUnexpected error in chat session: {e}\n")

if __name__ == "__main__":
    main()
