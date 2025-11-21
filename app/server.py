# app/server.py
from fastapi import FastAPI
from pydantic import BaseModel
from .model_loader import run_inference
from .monitor import get_system_stats
from .logger import log_metrics
import time

app = FastAPI(title="Adaptive Scheduler - Phi3 Mode")

class InferenceRequest(BaseModel):
    prompt: str


@app.post("/infer")
async def infer(req: InferenceRequest):

    selected_model = "gemma3"
    decision_reason = "Forced GPU gemma3 test mode"

    stats_before = get_system_stats()

    start = time.time()
    output, latency = run_inference(req.prompt, selected_model)
    end = time.time()

    stats_after = get_system_stats()

    # Deltas
    gpu_util_delta = stats_after["gpu_util"] - stats_before["gpu_util"]
    cpu_util_delta = stats_after["cpu_util"] - stats_before["cpu_util"]

    output_tokens = len(output.split())
    throughput = round(output_tokens / latency, 2) if latency > 0 else 0.0

    record = {
        "timestamp": stats_before["timestamp"],
        "mode": "gpu",
        "selected_model": selected_model,
        "decision_reason": decision_reason,
        "latency_s": round(latency, 3),
        "prompt_length": len(req.prompt.split()),
        "output_tokens": output_tokens,
        "throughput_tokens_per_s": throughput,

        "cpu_util_before": stats_before["cpu_util"],
        "cpu_util_after": stats_after["cpu_util"],
        "cpu_util_delta": cpu_util_delta,

        "gpu_util_before": stats_before["gpu_util"],
        "gpu_util_after": stats_after["gpu_util"],
        "gpu_util_delta": gpu_util_delta,

        "cpu_mem_before_gb": stats_before["cpu_mem_used_gb"],
        "cpu_mem_after_gb": stats_after["cpu_mem_used_gb"],

        "gpu_mem_before_gb": stats_before["gpu_mem_used_gb"],
        "gpu_mem_after_gb": stats_after["gpu_mem_used_gb"],

        "gpu_mem_util_before_pct": stats_before["gpu_mem_util_pct"],
        "gpu_mem_util_after_pct": stats_after["gpu_mem_util_pct"],
    }

    log_metrics(record)

    record["output"] = output[:1200]
    return record
