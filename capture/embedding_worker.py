import time
import threading
import numpy as np

class EmbeddingWorker(threading.Thread):
    """
    Background worker thread that polls the database for OCR-processed snapshots
    that do not yet have text embeddings, generates them using sentence-transformers,
    and inserts the vector embeddings into the virtual snapshot_vectors table.
    """
    def __init__(self, db_path: str, poll_interval: int, batch_size: int, model_name: str):
        super().__init__()
        self.db_path = db_path
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.model_name = model_name
        self.stop_event = threading.Event()
        self.daemon = True  # Ensure thread stops if main thread dies

    def run(self):
        # CRITICAL PERFORMANCE REQUIREMENT:
        # Load the SentenceTransformer model EXACTLY ONCE per worker lifetime, here on thread startup.
        # This operation is heavy (~35s loading weights) and must never run inside the polling loop.
        print(f"Embedding Worker: Loading model '{self.model_name}' (this may take up to 30-40 seconds)...")
        start_time = time.time()
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.model_name)
            load_duration = time.time() - start_time
            print(f"Embedding Worker: Model loaded successfully in {load_duration:.2f}s. Thread ready for tasks.")
        except Exception as e:
            print(f"Embedding Worker CRITICAL ERROR: Failed to load embedding model: {e}")
            return

        while not self.stop_event.is_set():
            try:
                # Poll and process a batch
                self._process_batch(model)
            except Exception as e:
                print(f"Error in Embedding Worker polling loop: {e}")
                
            # Sleep in small chunks to allow quick reaction to stop_event
            sleep_remaining = self.poll_interval
            while sleep_remaining > 0 and not self.stop_event.is_set():
                sleep_chunk = min(0.5, sleep_remaining)
                time.sleep(sleep_chunk)
                sleep_remaining -= sleep_chunk
                
        print("Embedding Worker stopped.")

    def _process_batch(self, model):
        from storage.db import get_unembedded_snapshots, insert_embedding
        
        # Poll database for snapshots with OCR text but no vector embedding
        rows = get_unembedded_snapshots(self.db_path, self.batch_size)
        if not rows:
            return

        for row in rows:
            if self.stop_event.is_set():
                break
                
            snapshot_id = row['id']
            ocr_text = row['ocr_text']
            
            try:
                # Generate 384-dimensional vector embedding
                embedding = model.encode(ocr_text)
                
                # Convert the float list/numpy array to float32 binary format
                # float32 matches float[384] database declaration and is highly compact
                vector_bytes = embedding.astype(np.float32).tobytes()
                
                # Insert the vector into the vec0 table
                insert_embedding(self.db_path, snapshot_id, vector_bytes)
                print(f"Embedding Worker: Successfully embedded snapshot ID {snapshot_id}")
                
            except Exception as e:
                # Handle failure per-snapshot gracefully: log it and continue.
                # Unlike OCR failures, we do not write an error placeholder inside the vector table
                # (since there is no logical vector representation for an error).
                # We leave the row unembedded in the DB so it will automatically be retried in future poll cycles.
                print(f"Embedding Worker: Failed to embed snapshot ID {snapshot_id}: {e}")

    def stop(self):
        """Signals the background loop to terminate gracefully."""
        self.stop_event.set()
