# Local Recall

> [!WARNING]
> **Project Status: Early Prototype under Active Development**  
> This repository is an early proof-of-concept and is under active development. APIs, database schemas, and workflows are subject to change.

Local Recall is a fully local, privacy-first alternative to Microsoft Recall. It runs entirely on your own machine, capturing screen snapshots, extracting text, generating vector embeddings, and allowing you to conversate with your screen history using a local Large Language Model (LLM).


---

## Technical Pipeline Architecture

Local Recall is built on an asynchronous pipeline structured across five key phases:

1. **Capture & Deduplication**: Captures screen snapshots every 5 seconds using `mss`. Compares the perceptual hash (`imagehash` + `Pillow`) of the current frame against the previous one. Static screens are skipped to save disk space and database footprint.
2. **Local OCR (Out-of-Band)**: A background worker (`ocr_worker.py`) polls the SQLite database for unprocessed snapshots and extracts text using Windows' built-in high-performance **WinRT OCR Engine** (`Windows.Media.Ocr`).
3. **Local Vector Embeddings**: A background worker (`embedding_worker.py`) extracts OCR texts, generates 384-dimensional vector representations using `sentence-transformers` (`all-MiniLM-L6-v2`), and serializes them to float32 byte buffers.
4. **Vector Database**: Stores vector embeddings in a virtual SQLite vector table (`snapshot_vectors`) powered by the lightweight **`sqlite-vec`** extension, enabling sub-50ms K-Nearest Neighbor (KNN) semantic searches locally.
5. **Retrieval-Augmented Generation (RAG)**: Integrates with a local LM Studio server running **Qwen 2.5-7B** to provide interactive conversational search over captured screen context with zero data leakage.

---

## Project Structure

```text
local-recall/
├── capture/
│   ├── capture_loop.py       # Manages screen grabbing and metadata logging
│   ├── ocr_worker.py         # Polling worker calling Windows.Media.Ocr
│   ├── embedding_worker.py   # Polling worker calling sentence-transformers
│   └── window_info.py        # Windows ctypes utility to grab process & app metadata
├── storage/
│   └── db.py                 # SQLite + sqlite-vec schema, migrations, and queries
├── templates/
│   └── index.html            # Clean light-themed HTML UI template
├── data/
│   ├── snapshots/            # Saved deduplicated screen captures (.webp)
│   └── localrecall.db        # SQLite database (metadata + vectors)
├── config.py                 # Global constants (polling rates, hosts, thresholds)
├── main.py                   # Main pipeline entry point (runs capture, OCR, & embeddings)
├── query.py                  # One-shot command line semantic search utility
├── chat.py                   # Conversational REPL CLI interface
├── app.py                    # Flask Web Explorer UI server
├── requirements.txt          # Pinned project dependencies
├── .env                      # User-configurable environment variables
└── README.md                 # Project documentation
```

---

## Installation & Setup

### Prerequisites
*   **Operating System**: Windows 10/11 (required for WinRT OCR and ctypes hooks).
*   **LM Studio**: Install [LM Studio](https://lmstudio.ai/) and start the local OpenAI-compatible server at `http://localhost:1234/v1` with a model (e.g. `Qwen/Qwen2.5-7B-Instruct-GGUF`) loaded.

### Installation
1. Clone the repository and navigate into it.
2. Create and activate a Python virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
3. Install project dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
4. Customize your configuration parameters by editing the `.env` file in the project root.

### Configuration Template (`.env`)
```ini
# --- Capture Loop ---
CAPTURE_INTERVAL_SECONDS=5
HASH_DISTANCE_THRESHOLD=8

# --- Background Worker Polling ---
OCR_POLL_INTERVAL_SECONDS=3
OCR_BATCH_SIZE=5
EMBEDDING_POLL_INTERVAL_SECONDS=3
EMBEDDING_BATCH_SIZE=5

# --- Sentence-Transformers Model ---
EMBEDDING_MODEL_NAME=your_embedding_model_name_here
EMBEDDING_DIMENSION=384

# --- Local LLM (LM Studio) ---
LM_STUDIO_BASE_URL=http://your_lm_studio_ip_or_host:port/v1
LM_STUDIO_MODEL_NAME=your_loaded_llm_model_id_here

# --- Flask Web Explorer UI ---
WEB_HOST=your_web_host_ip
WEB_PORT=your_web_host_port
```

---

## Running Local Recall

### 1. Run the Background Capture Pipeline
Start the background capture loop, OCR parser, and embedding worker:
```powershell
.\.venv\Scripts\python main.py
```
This logs snapshot captures, active application states, and background OCR/embedding progress to the console. Press `Ctrl+C` to terminate gracefully.

### 2. Search History via the CLI
Use the search tool to locate snapshots semantically matching a query string:
```powershell
.\.venv\Scripts\python query.py "Wikipedia Recall page" 3
```
This performs a KNN search on the vector table and prints the top 3 matches, L2 vector distance, and metadata info to the console.

### 3. Start a Conversational Q&A CLI session
Launch the interactive console assistant:
```powershell
.\.venv\Scripts\python chat.py
```
Type your questions (e.g. `what YouTube video did I watch?`). The tool embeds the prompt, queries the database context, and passes it to the local LLM. Inline sources (timestamps and titles) are printed beneath the answer.

### 4. Start the Web Explorer UI
Run the local Flask web explorer:
```powershell
.\.venv\Scripts\python app.py
```
Open `http://127.0.0.1:5000` in your web browser. This presents a simple light-themed UI to search your screen history, read conversational answers, and view grid previews of matching screenshots.
