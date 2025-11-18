# app/scheduler.py
def choose_model(stats: dict, prompt: str, force_mode: str = None) -> tuple[str, str]:
    """
    Adaptive rule-based scheduler that chooses between CPU and GPU models.
    
    Args:
        stats: System statistics (CPU/GPU utilization, memory)
        prompt: Input prompt text
        force_mode: Optional override ('cpu', 'gpu', or None for adaptive)
    
    Returns:
        tuple: (selected_model, decision_reason)
    """
    
    # Force GPU mode for testing
    if force_mode == "gpu":
        return "gemma3", "Force GPU mode - testing gemma3 on GPU"
    
    # Force CPU mode
    if force_mode == "cpu":
        return "phi3", "Force CPU mode - testing phi3 on CPU"
    
    # Adaptive scheduling (original logic)
    gpu_util = stats.get("gpu_util") or 0
    cpu_util = stats.get("cpu_util") or 0
    gpu_mem_used = stats.get("gpu_mem_used_gb") or 0
    gpu_mem_total = stats.get("gpu_mem_total_gb") or 1
    gpu_mem_ratio = gpu_mem_used / gpu_mem_total
    prompt_len = len(prompt.split())

    # Rule 1: GPU is both busy and nearly full
    if gpu_util > 85 and gpu_mem_ratio > 0.95:
        return "phi3", f"GPU overloaded (util={gpu_util}%, mem={gpu_mem_ratio*100:.1f}%)"

    # Rule 2: Large prompt → prefer GPU
    if prompt_len > 60 and gpu_util < 90:
        return "gemma3", f"Large prompt (len={prompt_len}) with available GPU"

    # Rule 3: Small prompt + low CPU load → use CPU
    if prompt_len < 30 and cpu_util < 80:
        return "phi3", f"Small prompt (len={prompt_len}) with low CPU load"

    # Rule 4: CPU overloaded → move to GPU
    if cpu_util > 75 and gpu_util < 80:
        return "gemma3", f"CPU overloaded (util={cpu_util}%), offloading to GPU"

    # Fallback: choose less busy resource
    if gpu_util < cpu_util:
        return "gemma3", f"GPU less busy (GPU={gpu_util}% vs CPU={cpu_util}%)"
    else:
        return "phi3", f"CPU less busy (CPU={cpu_util}% vs GPU={gpu_util}%)"