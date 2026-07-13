import os
import sys

# 1. Dependency check to guide users to run within the virtual environment
try:
    import sentence_transformers  # Load first to prevent DLL conflicts (WinError 1114) on Windows
    import mss
    import PIL
    import imagehash
except ImportError:
    print("=" * 70)
    print("ERROR: Missing required dependencies (sentence-transformers, mss, Pillow, or imagehash).")
    print("It looks like you are running Local Recall with the global Python interpreter")
    print("instead of the project's virtual environment.")
    print("\nTo run Local Recall correctly, use the virtual environment runner:")
    print("  .\\.venv\\Scripts\\python main.py")
    print("\nOr activate the environment first:")
    print("  .\\.venv\\Scripts\\activate")
    print("  python main.py")
    print("=" * 70)
    sys.exit(1)

import config
from storage.db import initialize_db
from capture.capture_loop import start_capture_loop
from capture.ocr_worker import OcrWorker
from capture.embedding_worker import EmbeddingWorker

def main():
    print("=" * 50)
    print("LOCAL RECALL - Startup")
    print("=" * 50)
    
    # 1. Initialize DB and directory structure (including migrations)
    try:
        initialize_db(config.DATABASE_PATH)
        os.makedirs(config.SNAPSHOTS_DIR, exist_ok=True)
    except Exception as e:
        print(f"Critical Error: Failed to initialize database or directory: {e}")
        sys.exit(1)
        
    # 2. Start OCR worker thread
    ocr_worker = OcrWorker(
        db_path=config.DATABASE_PATH,
        poll_interval=config.OCR_POLL_INTERVAL_SECONDS,
        batch_size=config.OCR_BATCH_SIZE
    )
    ocr_worker.start()
    
    # 3. Start Embedding worker thread
    embedding_worker = EmbeddingWorker(
        db_path=config.DATABASE_PATH,
        poll_interval=config.EMBEDDING_POLL_INTERVAL_SECONDS,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        model_name=config.EMBEDDING_MODEL_NAME
    )
    embedding_worker.start()
        
    # 4. Run the main capture loop
    try:
        start_capture_loop(
            db_path=config.DATABASE_PATH,
            snapshots_dir=config.SNAPSHOTS_DIR,
            interval=config.CAPTURE_INTERVAL_SECONDS,
            threshold=config.HASH_DISTANCE_THRESHOLD
        )
    except KeyboardInterrupt:
        print("\nShutdown signal received (Ctrl+C). Cleaning up resources...")
        print("Stopping background OCR worker...")
        ocr_worker.stop()
        print("Stopping background embedding worker...")
        embedding_worker.stop()
        ocr_worker.join(timeout=5)
        embedding_worker.join(timeout=5)
        print("Local Recall has shut down cleanly.")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error in main loop: {e}")
        print("Stopping background OCR worker...")
        ocr_worker.stop()
        print("Stopping background embedding worker...")
        embedding_worker.stop()
        ocr_worker.join(timeout=5)
        embedding_worker.join(timeout=5)
        sys.exit(1)


if __name__ == "__main__":
    main()

