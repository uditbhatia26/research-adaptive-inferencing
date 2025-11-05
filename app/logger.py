# app/logger.py
import csv
import os

LOG_FILE = "metrics_log.csv"

# Define consistent column order for all logs
LOG_COLUMNS = [
    "timestamp",
    "selected_model",
    "latency_s",
    "prompt_length",
    "output_tokens",
    "throughput_tokens_per_s",
    "cpu_util_before",
    "cpu_util_after",
    "cpu_util_delta",
    "gpu_util_before",
    "gpu_util_after",
    "gpu_util_delta",
    "cpu_mem_before_gb",
    "cpu_mem_after_gb",
    "gpu_mem_before_gb",
    "gpu_mem_after_gb",
    "gpu_mem_util_before_pct",
    "gpu_mem_util_after_pct"
]

def init_log_file():
    """Create log file with headers if it doesn't exist."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writeheader()

def log_metrics(data: dict):
    """Append a single inference record to CSV log."""
    init_log_file()  # ensures file exists with headers
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        # Only keep the keys that exist in our defined columns
        filtered = {k: data.get(k, None) for k in LOG_COLUMNS}
        writer.writerow(filtered)
