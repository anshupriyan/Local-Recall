import time
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import config

_model_instance = None

def get_embedding_model() -> SentenceTransformer:
    """
    Returns the SentenceTransformer embedding model, loading it exactly once on the first call (lazy singleton).
    """
    global _model_instance
    if _model_instance is None:
        print(f"Recall Engine: Loading embedding model '{config.EMBEDDING_MODEL_NAME}'...")
        start_time = time.time()
        _model_instance = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        print(f"Recall Engine: Embedding model loaded successfully in {time.time() - start_time:.2f}s.")
    return _model_instance

def build_context_block(results: list[dict]) -> str:
    """
    Formulates a formatted string context block from retrieved snapshots.
    Truncates OCR text to 500 characters to keep context size manageable.
    """
    context_parts = []
    for idx, row in enumerate(results):
        app = row['app_name'] or 'None'
        title = row['window_title'] or 'None'
        timestamp = row['timestamp']
        
        ocr_text = row['ocr_text'] or ""
        # Truncate long text
        if len(ocr_text) > 500:
            ocr_text = ocr_text[:500] + "... (truncated)"
            
        context_parts.append(
            f"--- Screenshot Reference {idx+1} ---\n"
            f"Timestamp: {timestamp}\n"
            f"App Name: {app}\n"
            f"Window Title: {title}\n"
            f"Captured Text: \"{ocr_text.strip()}\"\n"
        )
    return "\n".join(context_parts)

def query_local_llm(query_text: str, context_block: str) -> str:
    """
    Sends the context and question to the local LM Studio completions endpoint.
    """
    url = f"{config.LM_STUDIO_BASE_URL.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    system_prompt = (
        "You are Local Recall Assistant, a privacy-first AI that answers questions based on "
        "the user's captured screen history.\n"
        "Your task is to answer the user's question using ONLY the provided screen context. "
        "Strictly adhere to the following rules:\n\n"
        "1. SOURCE OF TRUTH: The ONLY valid source of truth is the metadata fields ('Window Title', 'App Name') "
        "and 'Captured Text' within each 'Screenshot Reference' block. If a piece of information appears in "
        "a 'Window Title' (e.g., a YouTube video title, a search query, a website page name), it is a fully valid "
        "and sufficient answer and must be used directly.\n"
        "2. INLINE CITATIONS: For every claim, fact, or answer you provide, you MUST explicitly cite the source "
        "Screenshot Reference number(s) inline in your response (e.g., '(Screenshot Reference 1)' or "
        "'(Screenshot Reference 3)'). Failure to cite is unacceptable.\n"
        "3. STRICT GROUNDING: Forbid stating any specific name, title, date, URL, or fact that does not literally "
        "and exactly appear in the provided Screenshot References. Do not make inferences, extrapolate, or fill in gaps "
        "with plausible details from your general knowledge. If it's not literally written in the context, it does not exist.\n"
        "4. FALLBACK: Say clearly \"Based on your screen history, I cannot find the answer to this question.\" ONLY if "
        "none of the Screenshot References contain any relevant information. Do not trigger this fallback if the answer is "
        "present in a metadata field (like 'Window Title') or inside the captured text."
    )
    
    user_prompt = f"Screen History Context:\n{context_block}\n\nQuestion: {query_text}"
    
    payload = {
        "model": config.LM_STUDIO_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return (
            "[CONNECTION ERROR] Could not connect to LM Studio at {url}.\n"
            "Please ensure LM Studio is running, its Local Server is started (port 1234), "
            "and model '{model}' is loaded."
        ).format(url=config.LM_STUDIO_BASE_URL, model=config.LM_STUDIO_MODEL_NAME)
    except Exception as e:
        return f"[ERROR] Request to LM Studio failed: {e}"
