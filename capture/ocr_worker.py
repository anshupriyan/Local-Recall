import os
import asyncio
import threading
import time

class OcrWorker(threading.Thread):
    """
    Background worker thread that polls the database for unprocessed screenshots,
    runs them through Windows.Media.Ocr via winrt, and updates the DB with results.

    Thread vs. Process Tradeoff:
    ---------------------------
    - Thread (Chosen):
      - Tradeoff: Shared memory space with main thread, extremely low overhead, 
        and fast startup. Since PyWinRT manages COM STA/MTA apartments natively 
        per thread, running a thread-specific asyncio event loop isolates the 
        async WinRT calls perfectly without needing process boundary marshalling.
      - SQLite connections are created and closed on-demand inside DB utility calls, 
        ensuring strict thread isolation (each thread uses its own DB connection).
    - Separate Process:
      - Tradeoff: Bypasses the Global Interpreter Lock (GIL) and isolates crashes. 
        However, since OCR takes ~430ms and runs at a low-frequency poll (every 3s), 
        CPU overhead is minimal, and the GIL is released during native WinRT OCR execution. 
        A thread is preferred to avoid the resource footprint of spawning child OS processes.
    """
    def __init__(self, db_path: str, poll_interval: int, batch_size: int):
        super().__init__()
        self.db_path = db_path
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.stop_event = threading.Event()
        self.daemon = True  # Ensure thread dies if main thread exits unexpectedly

    def run(self):
        # Create and set a thread-specific event loop for WinRT async calls
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._worker_loop())
        finally:
            loop.close()

    async def _worker_loop(self):
        print("OCR Background Worker started.")
        while not self.stop_event.is_set():
            try:
                # Process a batch of unprocessed snapshots
                processed_count = await self._process_batch()
                
                # Check stop event frequently during sleep period
                sleep_remaining = self.poll_interval
                while sleep_remaining > 0 and not self.stop_event.is_set():
                    sleep_chunk = min(0.5, sleep_remaining)
                    await asyncio.sleep(sleep_chunk)
                    sleep_remaining -= sleep_chunk
            except Exception as e:
                print(f"Error in OCR background worker loop: {e}")
                # Wait briefly before retrying if there's a loop-level failure
                await asyncio.sleep(self.poll_interval)
                
        print("OCR Background Worker stopped.")

    async def _process_batch(self) -> int:
        from storage.db import get_unprocessed_snapshots
        
        # Poll the database for snapshots without OCR text
        rows = get_unprocessed_snapshots(self.db_path, self.batch_size)
        if not rows:
            return 0
            
        for row in rows:
            if self.stop_event.is_set():
                break
                
            snapshot_id = row['id']
            filepath = row['filepath']
            
            await self._process_single_snapshot(snapshot_id, filepath)
            
        return len(rows)

    async def _process_single_snapshot(self, snapshot_id: int, filepath: str):
        from storage.db import update_ocr_text
        
        if not os.path.exists(filepath):
            error_msg = f"[ERROR] File not found: {filepath}"
            print(f"OCR Worker: {error_msg}")
            update_ocr_text(self.db_path, snapshot_id, error_msg)
            return

        try:
            # Lazy imports of winrt to keep thread initialization quick
            from winrt.windows.storage import StorageFile, FileAccessMode
            from winrt.windows.graphics.imaging import BitmapDecoder
            from winrt.windows.media.ocr import OcrEngine

            # 1. Open the file via WinRT API
            file = await StorageFile.get_file_from_path_async(os.path.abspath(filepath))
            
            # 2. Open read-only stream
            stream = await file.open_async(FileAccessMode.READ)
            
            # 3. Decode into SoftwareBitmap
            decoder = await BitmapDecoder.create_async(stream)
            software_bitmap = await decoder.get_software_bitmap_async()
            
            # 4. Initialize Windows OCR Engine using user profile languages
            engine = OcrEngine.try_create_from_user_profile_languages()
            if not engine:
                raise RuntimeError(
                    "OcrEngine could not be created. "
                    "Make sure a Windows language pack with OCR support is installed."
                )
                
            # 5. Perform OCR
            result = await engine.recognize_async(software_bitmap)
            extracted_text = result.text
            
            # Write results back to the database
            update_ocr_text(self.db_path, snapshot_id, extracted_text)
            print(f"OCR Worker: Successfully processed snapshot ID {snapshot_id} ({len(extracted_text)} chars)")
            
        except Exception as e:
            # Update the row with an error message to prevent infinite retries
            error_msg = f"[ERROR] OCR processing failed: {str(e)}"
            print(f"OCR Worker: Failed processing snapshot ID {snapshot_id}: {e}")
            try:
                update_ocr_text(self.db_path, snapshot_id, error_msg)
            except Exception as db_err:
                print(f"OCR Worker: Database write failed during error update: {db_err}")

    def stop(self):
        """Signals the background loop to terminate gracefully."""
        self.stop_event.set()
