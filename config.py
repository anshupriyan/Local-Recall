# Local Recall Configuration System
import os
from dotenv import load_dotenv

# Load user environment variables from .env file if it exists
load_dotenv()

# --- Capture settings ---
CAPTURE_INTERVAL_SECONDS = int(os.getenv("CAPTURE_INTERVAL_SECONDS", 5))
HASH_DISTANCE_THRESHOLD = int(os.getenv("HASH_DISTANCE_THRESHOLD", 8))

# --- Paths ---
DATA_DIR = "data"
DATABASE_PATH = os.path.join(DATA_DIR, "localrecall.db")
SNAPSHOTS_DIR = os.path.join(DATA_DIR, "snapshots")

# --- Background OCR configurations ---
OCR_POLL_INTERVAL_SECONDS = int(os.getenv("OCR_POLL_INTERVAL_SECONDS", 3))
OCR_BATCH_SIZE = int(os.getenv("OCR_BATCH_SIZE", 5))

# --- Background Embedding configurations ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 384))
EMBEDDING_POLL_INTERVAL_SECONDS = int(os.getenv("EMBEDDING_POLL_INTERVAL_SECONDS", 3))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 5))

# --- LM Studio Local LLM configurations ---
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL_NAME = os.getenv("LM_STUDIO_MODEL_NAME", "qwen2.5-7b-instruct")

# --- Local Web Interface configurations ---
WEB_HOST = os.getenv("WEB_HOST", "127.0.0.1")
WEB_PORT = int(os.getenv("WEB_PORT", 5000))
