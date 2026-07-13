import sys
import os
import time
import numpy as np

# Verify that sentence-transformers is accessible
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not found.")
    print("Please execute query.py using the project virtual environment runner:")
    print("  .\\.venv\\Scripts\\python query.py \"<your search query>\"")
    sys.exit(1)

import config
from storage.db import search_snapshots

def print_usage():
    print("Local Recall Semantic Search CLI")
    print("-" * 50)
    print("Usage:")
    print("  .\\.venv\\Scripts\\python query.py \"<search query>\" [top_k]")
    print("\nExamples:")
    print("  .\\.venv\\Scripts\\python query.py \"Wikipedia page about Windows Recall\"")
    print("  .\\.venv\\Scripts\\python query.py \"KeyboardInterrupt exit codes\" 5")
    print("-" * 50)

def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
        
    query_text = sys.argv[1].strip()
    if not query_text:
        print("ERROR: Empty query string provided.")
        print_usage()
        sys.exit(1)
        
    top_k = 3
    if len(sys.argv) >= 3:
        try:
            top_k = int(sys.argv[2])
            if top_k <= 0:
                raise ValueError()
        except ValueError:
            print(f"Warning: Invalid top_k value '{sys.argv[2]}'. Defaulting to 3.")
            top_k = 3
            
    db_path = config.DATABASE_PATH
    if not os.path.exists(db_path):
        print(f"ERROR: Database file '{db_path}' not found.")
        print("Please ensure the capture loop (main.py) has run and captured snapshots.")
        sys.exit(1)

    print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}'...")
    print("   (This takes about 30 seconds on cold start)...")
    start_load = time.time()
    try:
        model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"ERROR: Failed loading embedding model: {e}")
        sys.exit(1)
    print(f"   Model loaded successfully in {time.time() - start_load:.2f}s.\n")

    print(f"Searching for: \"{query_text}\" (returning top {top_k} matches)...")
    start_query = time.time()
    
    # 1. Embed query string
    query_vector = model.encode(query_text)
    
    # 2. Serialize vector embedding to float32 binary format
    query_vec_bytes = query_vector.astype(np.float32).tobytes()
    
    # 3. Query the virtual vector table (snapshot_vectors)
    try:
        results = search_snapshots(db_path, query_vec_bytes, top_k)
    except Exception as e:
        print(f"ERROR: Search execution failed: {e}")
        sys.exit(1)
        
    query_duration = time.time() - start_query
    print(f"Search completed in {query_duration * 1000:.2f}ms. Found {len(results)} matches:")
    print("=" * 85)
    
    if not results:
        print("No matching snapshots found.")
        print("=" * 85)
        sys.exit(0)
        
    for idx, row in enumerate(results):
        app = row['app_name'] or 'None'
        title = row['window_title'] or 'None'
        timestamp = row['timestamp']
        distance = row['distance']
        
        # Clean text snippet formatting
        snippet = "[No matching OCR text]"
        if row['ocr_text']:
            snippet = row['ocr_text'][:160].strip().replace('\n', ' ')
            if len(row['ocr_text']) > 160:
                snippet += "..."
                
        print(f"Rank {idx+1} | Snapshot ID: {row['id']} | Distance (L2): {distance:.4f}")
        print(f"  Captured:   {timestamp}")
        print(f"  App Name:   {app}")
        print(f"  Win Title:  {title}")
        print(f"  OCR Match:  \"{snippet}\"")
        print("-" * 85)

if __name__ == "__main__":
    main()
