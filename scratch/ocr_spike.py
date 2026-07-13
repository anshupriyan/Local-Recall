import os
import time
import asyncio
import glob
from PIL import Image, ImageDraw

# Import WinRT projections
from winrt.windows.storage import StorageFile, FileAccessMode
from winrt.windows.graphics.imaging import BitmapDecoder
from winrt.windows.media.ocr import OcrEngine

async def run_ocr_on_file(file_path: str) -> tuple[str, float]:
    """
    Loads an image file, decodes it into a Windows SoftwareBitmap,
    runs it through the Windows.Media.Ocr engine, and returns the result text
    along with the execution duration.
    """
    abs_path = os.path.abspath(file_path)
    
    start_time = time.time()
    
    # 1. Open the file as a WinRT StorageFile
    file = await StorageFile.get_file_from_path_async(abs_path)
    
    # 2. Open read-only stream
    stream = await file.open_async(FileAccessMode.READ)
    
    # 3. Create bitmap decoder and get the SoftwareBitmap object
    decoder = await BitmapDecoder.create_async(stream)
    software_bitmap = await decoder.get_software_bitmap_async()
    
    # 4. Try creating OcrEngine for user profile languages
    engine = OcrEngine.try_create_from_user_profile_languages()
    if not engine:
        raise RuntimeError(
            "Could not initialize Windows.Media.Ocr.OcrEngine. "
            "Please ensure a Windows language pack with OCR support is installed."
        )
    
    # 5. Run OCR recognition asynchronously
    result = await engine.recognize_async(software_bitmap)
    
    duration = time.time() - start_time
    return result.text, duration

async def main():
    print("=" * 60)
    print("Local Recall - Windows.Media.Ocr Performance Spike")
    print("=" * 60)
    
    # Search for snapshot files
    snapshot_dir = "data/snapshots"
    search_pattern = os.path.join(snapshot_dir, "*.*")
    files = glob.glob(search_pattern)
    
    image_files = [f for f in files if f.lower().endswith(('.webp', '.png', '.jpg', '.jpeg'))]
    
    target_file = None
    temp_file_created = False
    
    if image_files:
        # Select the most recently modified snapshot
        target_file = max(image_files, key=os.path.getmtime)
        print(f"Targeting most recent snapshot: {target_file}")
    else:
        print("No snapshots found in data/snapshots/.")
        print("Generating a temporary test image with text to prove out OCR...")
        os.makedirs(snapshot_dir, exist_ok=True)
        target_file = os.path.join(snapshot_dir, "temp_ocr_test.png")
        
        # Generate a PNG image containing clean mock text
        img = Image.new("RGB", (600, 150), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((30, 60), "Local Recall OCR Spike - Testing Windows WinRT OCR Engine!", fill=(0, 0, 0))
        img.save(target_file)
        temp_file_created = True
        print(f"Temporary image created at: {target_file}")
        
    try:
        # Run OCR
        extracted_text, duration = await run_ocr_on_file(target_file)
        
        print("\n" + "-" * 50)
        print("Extracted OCR Text:")
        print("-" * 50)
        print(extracted_text if extracted_text.strip() else "[No text detected]")
        print("-" * 50)
        print(f"Latency: {duration:.4f} seconds ({duration * 1000:.2f} ms)")
        print("-" * 50)
        
    except Exception as e:
        print(f"\n[ERROR] OCR Spike execution failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up temporary test image if one was created
        if temp_file_created and os.path.exists(target_file):
            try:
                os.remove(target_file)
                print("Temporary test image cleaned up.")
            except Exception as e:
                print(f"Warning: Failed to clean up temporary file {target_file}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
