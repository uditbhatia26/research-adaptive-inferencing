from fastapi import FastAPI
from pydantic import BaseModel
from .model_loader import run_inference
from .monitor import get_system_stats
from .scheduler import choose_model
from .logger import log_metrics
import time

app = FastAPI(title="Adaptive Scheduler - Phase 5 (Experiment Mode)")

class InferenceRequest(BaseModel):
    prompt: str
    mode: str | None = "adaptive"  # "cpu", "gpu", or "adaptive"

@app.post("/infer")
async def infer(req: InferenceRequest):
    # 1️⃣ Capture pre-inference stats
    stats_before = get_system_stats()

    # 2️⃣ Choose model based on mode
    if req.mode == "cpu":
        selected_model = "phi3"
        decision_reason = "Forced CPU-only mode"
    elif req.mode == "gpu":
        selected_model = "gemma3"
        decision_reason = "Forced GPU-only mode"
    else:
        selected_model = choose_model(stats_before, req.prompt)
        decision_reason = "Adaptive scheduling decision"

    # 3️⃣ Run inference
    start = time.time()
    output, latency = run_inference(req.prompt, selected_model)
    end = time.time()

    # 4️⃣ Capture post-inference stats
    stats_after = get_system_stats()

    # 5️⃣ Compute deltas & derived metrics
    gpu_util_delta = None
    cpu_util_delta = None
    if stats_before["gpu_util"] is not None and stats_after["gpu_util"] is not None:
        gpu_util_delta = stats_after["gpu_util"] - stats_before["gpu_util"]
    if stats_before["cpu_util"] is not None and stats_after["cpu_util"] is not None:
        cpu_util_delta = stats_after["cpu_util"] - stats_before["cpu_util"]

    output_tokens = len(output.split())
    throughput = round(output_tokens / latency, 2) if latency > 0 else None

    # 6️⃣ Prepare record for CSV + response
    record = {
        "timestamp": stats_before["timestamp"],
        "mode": req.mode,
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
        "gpu_mem_util_after_pct": stats_after["gpu_mem_util_pct"]
    }

    # 7️⃣ Log metrics
    log_metrics(record)

    # 8️⃣ Return response
    record["output"] = output[:1200]  # trim long responses
    return record
