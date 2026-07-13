import os
import time
import sqlite3
import sqlite_vec
import numpy as np
from sentence_transformers import SentenceTransformer

def run_embedding_spike():
    print("=" * 70)
    print("Local Recall - Embedding + sqlite-vec Search Spike")
    print("=" * 70)
    
    # 1. Model Loading
    print("1. Loading SentenceTransformer ('all-MiniLM-L6-v2')...")
    start_time = time.time()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model_load_time = time.time() - start_time
    print(f"   [OK] Model loaded in {model_load_time:.4f} seconds ({model_load_time * 1000:.2f} ms)")
    
    # 2. Load Snapshots from localrecall.db
    real_db_path = "data/localrecall.db"
    if not os.path.exists(real_db_path):
        print(f"Error: {real_db_path} does not exist. Please run main.py first to populate snapshots.")
        return
        
    print(f"2. Pulling snapshots from {real_db_path}...")
    conn = sqlite3.connect(real_db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Fetch snapshots with non-NULL, non-error OCR texts
    cursor.execute("""
        SELECT id, filepath, window_title, app_name, ocr_text 
        FROM snapshots 
        WHERE ocr_text IS NOT NULL AND ocr_text NOT LIKE '[ERROR]%'
        LIMIT 10
    """)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("   No valid processed snapshots found in database.")
        return
        
    print(f"   Fetched {len(rows)} snapshots.")
    for row in rows:
        app = row['app_name'] or 'Unknown'
        title = row['window_title'] or 'Unknown'
        text_snippet = row['ocr_text'][:60].replace('\n', ' ') + "..."
        print(f"     - ID {row['id']}: [{app} | {title}] -> \"{text_snippet}\"")
        
    # 3. Initialize throwaway sqlite-vec database
    scratch_db_path = "data/test_vec_spike.db"
    if os.path.exists(scratch_db_path):
        os.remove(scratch_db_path)
        
    print(f"\n3. Initializing scratch database {scratch_db_path} and loading sqlite-vec extension...")
    vec_conn = sqlite3.connect(scratch_db_path)
    vec_conn.enable_load_extension(True)
    try:
        sqlite_vec.load(vec_conn)
        vec_conn.enable_load_extension(False)
        print("   [OK] Extension loaded successfully.")
    except Exception as e:
        print(f"   [ERROR] Failed to load sqlite-vec: {e}")
        vec_conn.close()
        return

    # Verify vec version
    version = vec_conn.execute("select vec_version()").fetchone()[0]
    print(f"   sqlite-vec version: {version}")
    
    # Create the virtual table for 384 dimensions (using L2 Euclidean distance metric by default)
    vec_conn.execute("""
        CREATE VIRTUAL TABLE vec_snapshots USING vec0(
            snapshot_id integer primary key,
            embedding float[384]
        )
    """)
    vec_conn.commit()
    print("   [OK] Virtual table vec_snapshots (384 dimensions) created.")

    # 4. Generate Embeddings & Insert
    print("\n4. Embedding OCR texts and inserting into virtual table...")
    embedding_durations = []
    
    for row in rows:
        snapshot_id = row['id']
        ocr_text = row['ocr_text']
        
        # Measure single embedding latency
        embed_start = time.time()
        embedding = model.encode(ocr_text)
        embed_duration = time.time() - embed_start
        embedding_durations.append(embed_duration)
        
        # Serialize the embedding vector to float32 binary format
        vec_bytes = embedding.astype(np.float32).tobytes()
        
        # Insert into virtual table
        vec_conn.execute(
            "INSERT INTO vec_snapshots(snapshot_id, embedding) VALUES (?, ?)",
            (snapshot_id, vec_bytes)
        )
        
    vec_conn.commit()
    
    avg_embed_time = sum(embedding_durations) / len(embedding_durations)
    print(f"   [OK] Processed {len(rows)} embeddings.")
    print(f"   Average single embedding time: {avg_embed_time:.4f} seconds ({avg_embed_time * 1000:.2f} ms)")
    
    # 5. Run Semantic Search Query
    # Choose a search phrase that matches one of the apps or topics from the pulled rows.
    # Let's inspect the pulled rows and query something relevant.
    # We will run two queries: one for "wikipedia recall article" and one for "keyboard interrupt exit code".
    search_queries = [
        "Wikipedia page about Windows Recall",
        "keyboard interrupt exit code shutdown"
    ]
    
    print("\n5. Running Semantic Search Queries...")
    
    # Reload metadata for printing results
    metadata_map = {row['id']: row for row in rows}
    
    for query_text in search_queries:
        print(f"\n   Query: '{query_text}'")
        
        # Embed query text
        query_start = time.time()
        query_vector = model.encode(query_text)
        query_vec_bytes = query_vector.astype(np.float32).tobytes()
        
        # Query nearest neighbors using MATCH
        # k = 3 retrieves top 3 neighbors
        cursor = vec_conn.cursor()
        cursor.execute("""
            SELECT snapshot_id, distance
            FROM vec_snapshots
            WHERE embedding MATCH ? AND k = 3
            ORDER BY distance
        """, (query_vec_bytes,))
        results = cursor.fetchall()
        query_duration = time.time() - query_start
        
        print(f"   Query took: {query_duration:.4f} seconds ({query_duration * 1000:.2f} ms)")
        print(f"   Top Results:")
        for idx, (snapshot_id, distance) in enumerate(results):
            meta = metadata_map.get(snapshot_id)
            if meta:
                app = meta['app_name'] or 'None'
                title = meta['window_title'] or 'None'
                print(f"     Rank {idx+1}: Snapshot ID {snapshot_id} (Distance: {distance:.4f})")
                print(f"             App: {app} | Title: {title}")
            else:
                print(f"     Rank {idx+1}: Snapshot ID {snapshot_id} (Distance: {distance:.4f})")

    # Clean up
    vec_conn.close()
    if os.path.exists(scratch_db_path):
        os.remove(scratch_db_path)
    print("\nCleanup completed.")
    print("=" * 70)

if __name__ == "__main__":
    run_embedding_spike()
