# app/model_loader.py
import requests
import time

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "gemma3"

def run_inference(prompt: str, model: str = DEFAULT_MODEL):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    start = time.time()
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        latency = time.time() - start
        response.raise_for_status()
        data = response.json()
        text = data.get("response", "").strip()
        return text, latency
    except Exception as e:
        latency = time.time() - start
        return f"[Error contacting Ollama: {e}]", latency
