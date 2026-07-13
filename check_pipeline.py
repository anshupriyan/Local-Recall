"""
Local Recall — pipeline health check.

Run this any time to see whether capture, OCR, and embedding are keeping up
with each other, plus basic storage-growth numbers.

Usage (from project root, with venv active or using .venv\\Scripts\\python):
    .venv\\Scripts\\python check_pipeline.py
"""

import os
import sqlite3
import sys
from datetime import datetime

import sqlite_vec

DB_PATH = "data/localrecall.db"
SNAPSHOTS_DIR = "data/snapshots"


def human_size(num_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def main():
    if not os.path.exists(DB_PATH):
        print(f"No database found at {DB_PATH} — has main.py been run yet?")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    c = conn.cursor()

    print("=" * 60)
    print("LOCAL RECALL — Pipeline Health Check")
    print("=" * 60)

    # --- Counts at each stage ---
    c.execute("SELECT COUNT(*) FROM snapshots")
    total = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM snapshots WHERE ocr_text IS NOT NULL")
    ocr_done = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM snapshots WHERE ocr_text LIKE '[ERROR]%'")
    ocr_errors = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM snapshot_vectors")
    embedded = c.fetchone()[0]

    print(f"\nTotal snapshots captured:   {total}")
    print(f"OCR processed:              {ocr_done} ({ocr_done - ocr_errors} ok, {ocr_errors} errors)")
    print(f"Embedded (searchable):      {embedded}")

    ocr_backlog = total - ocr_done
    embed_backlog = ocr_done - ocr_errors - embedded
    if ocr_backlog > 0:
        print(f"  -> OCR backlog: {ocr_backlog} snapshots waiting")
    if embed_backlog > 0:
        print(f"  -> Embedding backlog: {embed_backlog} snapshots waiting")
    if ocr_backlog == 0 and embed_backlog <= 0:
        print("  -> Pipeline is fully caught up. ✅")

    # --- Oldest / newest unprocessed ---
    c.execute(
        "SELECT MIN(timestamp), MAX(timestamp) FROM snapshots WHERE ocr_text IS NULL"
    )
    row = c.fetchone()
    if row and row[0]:
        print(f"\nOldest un-OCR'd snapshot:   {row[0]}")
        print(f"Newest un-OCR'd snapshot:   {row[1]}")

    # --- Recent errors, if any ---
    if ocr_errors > 0:
        print(f"\nMost recent OCR errors (up to 5):")
        c.execute(
            "SELECT id, timestamp, ocr_text FROM snapshots "
            "WHERE ocr_text LIKE '[ERROR]%' ORDER BY timestamp DESC LIMIT 5"
        )
        for sid, ts, err in c.fetchall():
            print(f"  ID {sid} ({ts}): {err}")

    # --- Storage growth ---
    db_size = os.path.getsize(DB_PATH)
    snap_size = 0
    snap_count_files = 0
    if os.path.isdir(SNAPSHOTS_DIR):
        for f in os.listdir(SNAPSHOTS_DIR):
            fp = os.path.join(SNAPSHOTS_DIR, f)
            if os.path.isfile(fp):
                snap_size += os.path.getsize(fp)
                snap_count_files += 1

    print(f"\nDatabase size:              {human_size(db_size)}")
    print(f"Snapshots folder:           {human_size(snap_size)} ({snap_count_files} files)")
    print(f"Total on-disk footprint:    {human_size(db_size + snap_size)}")
    if total > 0:
        print(f"Avg size per snapshot:      {human_size(snap_size / max(snap_count_files, 1))}")

    # --- App breakdown (top 5 apps by snapshot count) ---
    print(f"\nTop apps by snapshot count:")
    c.execute(
        "SELECT app_name, COUNT(*) as cnt FROM snapshots "
        "GROUP BY app_name ORDER BY cnt DESC LIMIT 5"
    )
    for app, cnt in c.fetchall():
        print(f"  {app or 'Unknown'}: {cnt}")

    conn.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()