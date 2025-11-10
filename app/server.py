from fastapi import FastAPI
from pydantic import BaseModel
from .model_loader import run_inference
from .monitor import get_system_stats
from .scheduler import choose_model
from .logger import log_metrics
import time

app = FastAPI(title="Adaptive Scheduler - Robust Metrics Logging")

class InferenceRequest(BaseModel):
    prompt: str
    mode: str | None = "adaptive"  # "cpu", "gpu", or "adaptive"
    model: str | None = None        # Optional explicit model control


@app.post("/infer")
async def infer(req: InferenceRequest):
    # 1️⃣ Capture pre-inference stats
    stats_before = get_system_stats()

    # 2️⃣ Select model
    if req.model:
        selected_model = req.model
        decision_reason = f"Explicit model selection: {req.model}"
    elif req.mode == "cpu":
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

    # 5️⃣ Compute deltas safely
    gpu_util_delta = (stats_after.get("gpu_util", 0) or 0) - (stats_before.get("gpu_util", 0) or 0)
    cpu_util_delta = (stats_after.get("cpu_util", 0) or 0) - (stats_before.get("cpu_util", 0) or 0)

    # 6️⃣ Compute throughput
    output_tokens = len(output.split())
    throughput = round(output_tokens / latency, 2) if latency > 0 else 0.0

    # 7️⃣ Prepare record for CSV
    record = {
        "timestamp": stats_before.get("timestamp"),
        "mode": req.mode,
        "selected_model": selected_model,
        "decision_reason": decision_reason,
        "latency_s": round(latency, 3),
        "prompt_length": len(req.prompt.split()),
        "output_tokens": output_tokens,
        "throughput_tokens_per_s": throughput,
        "cpu_util_before": stats_before.get("cpu_util", 0),
        "cpu_util_after": stats_after.get("cpu_util", 0),
        "cpu_util_delta": cpu_util_delta,
        "gpu_util_before": stats_before.get("gpu_util", 0),
        "gpu_util_after": stats_after.get("gpu_util", 0),
        "gpu_util_delta": gpu_util_delta,
        "cpu_mem_before_gb": stats_before.get("cpu_mem_used_gb", 0),
        "cpu_mem_after_gb": stats_after.get("cpu_mem_used_gb", 0),
        "gpu_mem_before_gb": stats_before.get("gpu_mem_used_gb", 0),
        "gpu_mem_after_gb": stats_after.get("gpu_mem_used_gb", 0),
        "gpu_mem_util_before_pct": stats_before.get("gpu_mem_util_pct", 0),
        "gpu_mem_util_after_pct": stats_after.get("gpu_mem_util_pct", 0),
    }

    # 8️⃣ Log metrics
    log_metrics(record)

    # 9️⃣ Return response
    record["output"] = output[:1200]  # trim long responses for API output
    return record
