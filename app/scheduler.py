# app/scheduler.py
def choose_model(stats: dict, prompt: str) -> str:
    """
    Adaptive rule-based scheduler:
    Chooses between CPU-friendly and GPU-friendly models based on
    system load (CPU/GPU utilization) and workload (prompt length).
    """

    gpu_util = stats.get("gpu_util") or 0
    cpu_util = stats.get("cpu_util") or 0
    gpu_mem_used = stats.get("gpu_mem_used") or 0
    gpu_mem_total = stats.get("gpu_mem_total") or 1
    gpu_mem_ratio = gpu_mem_used / gpu_mem_total
    prompt_len = len(prompt.split())

    # --- Revised Adaptive Heuristics ---

    # Rule 1: Use GPU unless it's *both* busy and full
    if gpu_util > 85 and gpu_mem_ratio > 0.95:
        return "phi3"

    # Rule 2: Large prompt → GPU
    if prompt_len > 60 and gpu_util < 90:
        return "gemma3"

    # Rule 3: Small prompt + low CPU load → CPU
    if prompt_len < 30 and cpu_util < 80:
        return "phi3"

    # Rule 4: CPU overloaded → move to GPU
    if cpu_util > 75 and gpu_util < 80:
        return "gemma3"

    # Fallback
    return "gemma3" if gpu_util < cpu_util else "phi3"
