# app/server.py
from fastapi import FastAPI
from pydantic import BaseModel
from .model_loader import run_inference
from .monitor import get_system_stats
from .scheduler import choose_model
import time

app = FastAPI(title="Adaptive Scheduler - Phase 3b (Detailed Metrics)")

class InferenceRequest(BaseModel):
    prompt: str


@app.post("/infer")
async def infer(req: InferenceRequest):
    # 1️⃣ Capture pre-inference metrics
    stats_before = get_system_stats()

    # 2️⃣ Choose model adaptively
    selected_model = choose_model(stats_before, req.prompt)

    # 3️⃣ Run inference
    start = time.time()
    output, latency = run_inference(req.prompt, selected_model)
    end = time.time()

    # 4️⃣ Capture post-inference metrics
    stats_after = get_system_stats()

    # 5️⃣ Compute derived metrics
    gpu_util_delta = None
    cpu_util_delta = None
    if stats_before["gpu_util"] is not None and stats_after["gpu_util"] is not None:
        gpu_util_delta = stats_after["gpu_util"] - stats_before["gpu_util"]
    if stats_before["cpu_util"] is not None and stats_after["cpu_util"] is not None:
        cpu_util_delta = stats_after["cpu_util"] - stats_before["cpu_util"]

    # Approximate throughput (tokens/sec estimate)
    output_tokens = len(output.split())
    throughput = round(output_tokens / latency, 2) if latency > 0 else None

    # 6️⃣ Return all stats
    response = {
        "timestamp": stats_before["timestamp"],
        "selected_model": selected_model,
        "latency_s": round(latency, 3),
        "prompt_length": len(req.prompt.split()),
        "output_tokens": output_tokens,
        "throughput_tokens_per_s": throughput,

        # CPU / GPU Utilization
        "cpu_util_before": stats_before["cpu_util"],
        "cpu_util_after": stats_after["cpu_util"],
        "cpu_util_delta": cpu_util_delta,
        "gpu_util_before": stats_before["gpu_util"],
        "gpu_util_after": stats_after["gpu_util"],
        "gpu_util_delta": gpu_util_delta,

        # Memory info
        "cpu_mem_before_gb": stats_before["cpu_mem_used_gb"],
        "cpu_mem_after_gb": stats_after["cpu_mem_used_gb"],
        "gpu_mem_before_gb": stats_before["gpu_mem_used_gb"],
        "gpu_mem_after_gb": stats_after["gpu_mem_used_gb"],

        # GPU Memory ratio
        "gpu_mem_util_before_pct": stats_before["gpu_mem_util_pct"],
        "gpu_mem_util_after_pct": stats_after["gpu_mem_util_pct"],

        "output": output[:1500]  # truncate long outputs for readability
    }

    return response
