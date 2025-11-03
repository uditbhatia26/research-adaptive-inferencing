# app/server.py
from fastapi import FastAPI
from pydantic import BaseModel
from .model_loader import run_inference

app = FastAPI(title="Adaptive Scheduler - Phase 1 (Ollama LLM)")

class InferenceRequest(BaseModel):
    prompt: str
    model: str | None = "gemma3"

@app.post("/infer")
async def infer(req: InferenceRequest):
    """
    Receives a prompt and optional model name, runs inference on Ollama,
    and returns the full generated response with latency.
    """
    output, latency = run_inference(req.prompt, req.model)
    return {
        "model": req.model,
        "latency_s": latency,
        "output": output
    }
