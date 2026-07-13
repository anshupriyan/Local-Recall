import sqlite3
import os
import datetime
import sqlite_vec
import config

def _connect_vec(db_path: str) -> sqlite3.Connection:
    """
    Helper to establish connection and load sqlite-vec extension.
    """
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn

def initialize_db(db_path: str):
    """
    Initializes the SQLite database, creates the snapshots and snapshot_vectors tables,
    and runs schema migrations.
    """
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    # Use _connect_vec so we can create/verify virtual tables using vec0
    conn = _connect_vec(db_path)
    try:
        cursor = conn.cursor()
        # 1. Create base metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filepath TEXT NOT NULL,
                phash TEXT NOT NULL,
                window_title TEXT,
                app_name TEXT
            )
        """)
        
        # 2. Check if ocr_text column exists (migration path)
        cursor.execute("PRAGMA table_info(snapshots)")
        columns = [row[1] for row in cursor.fetchall()]
        if "ocr_text" not in columns:
            cursor.execute("ALTER TABLE snapshots ADD COLUMN ocr_text TEXT DEFAULT NULL")
            
        # 3. Create virtual table for embeddings (float[384])
        cursor.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS snapshot_vectors USING vec0(
                snapshot_id integer primary key,
                embedding float[{config.EMBEDDING_DIMENSION}]
            )
        """)
        
        conn.commit()
    finally:
        conn.close()

def insert_snapshot(db_path: str, timestamp: str, filepath: str, phash: str, window_title: str | None, app_name: str | None) -> int:
    """
    Inserts a snapshot metadata row into the database. Returns the row ID of the inserted snapshot.
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO snapshots (timestamp, filepath, phash, window_title, app_name)
            VALUES (?, ?, ?, ?, ?)
        """, (timestamp, filepath, phash, window_title, app_name))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()

def get_all_snapshots(db_path: str) -> list[dict]:
    """
    Retrieves all snapshot records from the database ordered by timestamp descending.
    Returns a list of dicts mapping column names to values.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, filepath, phash, window_title, app_name, ocr_text FROM snapshots ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

def delete_snapshot(db_path: str, snapshot_id: int) -> bool:
    """
    Deletes a snapshot record and its vector from the database by ID and deletes the associated image file from disk.
    Returns True if the record was successfully deleted, False otherwise.
    """
    conn = _connect_vec(db_path)
    try:
        cursor = conn.cursor()
        # Fetch file path first to delete the file
        cursor.execute("SELECT filepath FROM snapshots WHERE id = ?", (snapshot_id,))
        row = cursor.fetchone()
        if not row:
            return False
        
        filepath = row[0]
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError as e:
                print(f"Warning: Failed to delete file {filepath}: {e}")

        # Delete database rows from both tables
        cursor.execute("DELETE FROM snapshots WHERE id = ?", (snapshot_id,))
        cursor.execute("DELETE FROM snapshot_vectors WHERE snapshot_id = ?", (snapshot_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()

def delete_older_than(db_path: str, days: int) -> int:
    """
    Deletes snapshots older than the specified number of days (based on their timestamp).
    Deletes both database records (metadata and vectors) and their corresponding image files.
    Returns the number of deleted snapshots.
    """
    cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
    cutoff_str = cutoff.isoformat()

    conn = _connect_vec(db_path)
    try:
        cursor = conn.cursor()
        # Find files to delete
        cursor.execute("SELECT id, filepath FROM snapshots WHERE timestamp < ?", (cutoff_str,))
        rows = cursor.fetchall()
        
        deleted_count = 0
        for snapshot_id, filepath in rows:
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError as e:
                    print(f"Warning: Failed to delete file {filepath} during purge: {e}")
            cursor.execute("DELETE FROM snapshots WHERE id = ?", (snapshot_id,))
            cursor.execute("DELETE FROM snapshot_vectors WHERE snapshot_id = ?", (snapshot_id,))
            deleted_count += 1
            
        conn.commit()
        return deleted_count
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_unprocessed_snapshots(db_path: str, limit: int) -> list[dict]:
    """
    Retrieves the oldest unprocessed snapshots (where ocr_text is NULL).
    Returns a list of dicts mapping column names to values.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, filepath 
            FROM snapshots 
            WHERE ocr_text IS NULL 
            ORDER BY timestamp ASC 
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

def update_ocr_text(db_path: str, snapshot_id: int, text: str | None) -> bool:
    """
    Updates the ocr_text column of a snapshot.
    Returns True if successful, False otherwise.
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE snapshots 
            SET ocr_text = ? 
            WHERE id = ?
        """, (text, snapshot_id))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()

def get_unembedded_snapshots(db_path: str, limit: int) -> list[dict]:
    """
    Retrieves snapshots that have valid OCR text but no embedding in snapshot_vectors yet.
    Returns the oldest first.
    """
    conn = _connect_vec(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.id, s.ocr_text 
            FROM snapshots s
            LEFT JOIN snapshot_vectors v ON s.id = v.snapshot_id
            WHERE s.ocr_text IS NOT NULL 
              AND s.ocr_text NOT LIKE '[ERROR]%' 
              AND v.snapshot_id IS NULL
            ORDER BY s.timestamp ASC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

def insert_embedding(db_path: str, snapshot_id: int, vector_bytes: bytes):
    """
    Inserts or replaces an embedding vector inside the snapshot_vectors virtual table.
    """
    conn = _connect_vec(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO snapshot_vectors(snapshot_id, embedding)
            VALUES (?, ?)
        """, (snapshot_id, vector_bytes))
        conn.commit()
    finally:
        conn.close()

def search_snapshots(db_path: str, query_vec_bytes: bytes, top_k: int) -> list[dict]:
    """
    Performs K-Nearest Neighbor (KNN) semantic search against the snapshot_vectors table,
    joining the matching records back to the metadata table.
    """
    conn = _connect_vec(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.id, s.timestamp, s.filepath, s.window_title, s.app_name, s.ocr_text, v.distance
            FROM snapshot_vectors v
            JOIN snapshots s ON v.snapshot_id = s.id
            WHERE v.embedding MATCH ? AND v.k = ?
            ORDER BY v.distance ASC
        """, (query_vec_bytes, top_k))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

