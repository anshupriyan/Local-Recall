import os
import sys
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
import sqlite3

# Ensure dependencies are available before launching Flask
try:
    import config
    from storage.db import search_snapshots
    from recall_engine import get_embedding_model, build_context_block, query_local_llm
except ImportError as e:
    print(f"ERROR: Missing modules. Run within the project virtual environment: {e}")
    sys.exit(1)

app = Flask(__name__)

db_path = config.DATABASE_PATH
if not os.path.exists(db_path):
    print(f"ERROR: Database file '{db_path}' not found. Please run main.py first.")
    sys.exit(1)

# Load embedding model ONCE at Flask application startup
print("Initializing Web UI application engine...")
try:
    embedding_model = get_embedding_model()
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load embedding model: {e}")
    sys.exit(1)

def get_snapshot_path(snapshot_id: int) -> str:
    """Helper to query the snapshot filepath directly from the SQLite database."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT filepath FROM snapshots WHERE id = ?", (snapshot_id,))
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        conn.close()

@app.route("/")
def index():
    """Serves the main Explorer single-page UI."""
    return render_template("index.html")

@app.route("/snapshot/<int:snapshot_id>")
def serve_snapshot(snapshot_id):
    """Retrieves a snapshot record's file path by ID and serves the raw .webp image."""
    filepath = get_snapshot_path(snapshot_id)
    if not filepath:
        return "Snapshot record not found in database", 404
        
    # Resolve relative paths relative to current working directory
    if not os.path.isabs(filepath):
        filepath = os.path.abspath(os.path.join(os.getcwd(), filepath))
        
    if not os.path.exists(filepath):
        return f"Snapshot file not found on disk at: {filepath}", 404
        
    return send_file(filepath, mimetype="image/webp")

@app.route("/ask", methods=["POST"])
def ask():
    """
    POST route that processes user query strings, performs semantic nearest-neighbor search,
    packages the context, and queries the local LLM. Returns JSON results.
    """
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400
        
    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Empty question provided"}), 400
        
    try:
        # 1. Generate query embedding
        query_vector = embedding_model.encode(question)
        query_vec_bytes = query_vector.astype(np.float32).tobytes()
        
        # 2. Query nearest matches from database
        results = search_snapshots(db_path, query_vec_bytes, top_k=5)
        
        if not results:
            return jsonify({
                "answer": "No relevant snapshots found in your history.",
                "sources": [],
                "error": None
            })
            
        # 3. Construct context block from references
        context_block = build_context_block(results)
        
        # 4. Request answer from local LLM
        answer = query_local_llm(question, context_block)
        
        # Detect if the query_local_llm returned a ConnectionError wrapper string
        error_msg = None
        if "[CONNECTION ERROR]" in answer:
            error_msg = (
                "LM Studio is not reachable. Showing search matches, but generated answer is unavailable.\n"
                "Please make sure LM Studio is running, its Local Server is started, and model "
                f"'{config.LM_STUDIO_MODEL_NAME}' is loaded."
            )
            
        return jsonify({
            "answer": answer,
            "sources": results,
            "error": error_msg
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to process query: {str(e)}"}), 500

if __name__ == "__main__":
    print(f"Starting Local Recall web server at http://{config.WEB_HOST}:{config.WEB_PORT}")
    app.run(host=config.WEB_HOST, port=config.WEB_PORT, debug=False)
