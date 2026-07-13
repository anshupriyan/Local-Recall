import os
import time
import datetime
import mss
from PIL import Image
import imagehash

from capture.window_info import get_active_window_info
from storage.db import insert_snapshot

def start_capture_loop(db_path: str, snapshots_dir: str, interval: int, threshold: int):
    """
    Runs the main capture loop. Captures the screen every `interval` seconds,
    compares it with the previous capture's perceptual hash, and if different
    enough, saves the screenshot and updates the SQLite metadata database.
    """
    # Ensure snapshots directory exists
    os.makedirs(snapshots_dir, exist_ok=True)
    
    print(f"Starting Local Recall Capture Loop:")
    print(f"  - Database: {os.path.abspath(db_path)}")
    print(f"  - Snapshot Dir: {os.path.abspath(snapshots_dir)}")
    print(f"  - Capture Interval: {interval}s")
    print(f"  - Hash Distance Threshold: {threshold} (lower = more similar)")
    print("Press Ctrl+C to terminate gracefully.")
    print("-" * 50)

    prev_hash = None
    
    with mss.mss() as sct:
        # sct.monitors[0] is the bounding box of all monitors.
        # sct.monitors[1] is the primary monitor.
        # If multiple monitors exist, we capture the primary monitor.
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        
        while True:
            start_time = time.time()
            try:
                # Capture the screen
                sct_img = sct.grab(monitor)
                
                # Convert raw BGRA bytes to PIL Image (RGB)
                # BGRX is used because mss returns BGRA pixel format where A is padding
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                
                # Compute 64-bit perceptual hash (phash)
                current_hash = imagehash.phash(img)
                
                should_save = True
                distance = None
                
                if prev_hash is not None:
                    distance = current_hash - prev_hash
                    if distance < threshold:
                        should_save = False

                if should_save:
                    # Retrieve window title and app name at the moment of capture
                    window_title, app_name = get_active_window_info()
                    
                    now = datetime.datetime.now()
                    timestamp_str = now.isoformat()
                    
                    # Generate filename (replace colons with dashes for OS compatibility)
                    filename = now.strftime("%Y-%m-%dT%H-%M-%S") + ".webp"
                    filepath = os.path.join(snapshots_dir, filename)
                    
                    # Save image as WebP (with compression to optimize disk usage)
                    img.save(filepath, "WEBP", quality=80)
                    
                    # Insert metadata row into DB
                    hash_str = str(current_hash)
                    snapshot_id = insert_snapshot(
                        db_path,
                        timestamp_str,
                        filepath,
                        hash_str,
                        window_title,
                        app_name
                    )
                    
                    dist_msg = f" (hash distance: {distance})" if distance is not None else " (initial)"
                    print(f"[{now.strftime('%H:%M:%S')}] Saved Snapshot ID {snapshot_id}{dist_msg}")
                    print(f"  App: {app_name or 'None'} | Title: {window_title or 'None'}")
                    
                    # Update previous hash
                    prev_hash = current_hash
                else:
                    # Screen is too similar, skip saving
                    pass
                    
            except Exception as e:
                print(f"Error in capture iteration: {e}")
            
            # Calculate elapsed time and adjust sleep duration to maintain precise interval
            elapsed = time.time() - start_time
            sleep_time = max(0.1, interval - elapsed)
            time.sleep(sleep_time)
