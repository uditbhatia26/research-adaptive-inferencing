import requests
import time

API_URL = "http://127.0.0.1:8000/infer"

# ✅ Define test prompts (short, medium, long)
TEST_PROMPTS = [
    "What is AI?",
    "Explain how photosynthesis works.",
    "Describe the architecture of convolutional neural networks in deep learning.",
    "Explain in detail how large language models like GPT or Gemini are trained and optimized to handle complex reasoning tasks, multi-step instructions, and contextual understanding over long conversations. Discuss the underlying architecture such as the Transformer, including how attention mechanisms work to allow models to focus on relevant parts of the input sequence. Then describe the training process at scale — including dataset preparation, tokenization, unsupervised pretraining, supervised fine-tuning, and reinforcement learning from human feedback (RLHF). Explain how these models manage memory, gradient accumulation, and distributed training across thousands of GPUs using techniques like data parallelism, model parallelism, and mixed-precision training. Elaborate on the role of optimization algorithms such as AdamW, learning rate schedules, and checkpointing in stabilizing convergence. After that, discuss the major challenges that arise when deploying such models for real-world inference — like latency, energy consumption, and hardware constraints. Describe how quantization, pruning, and distillation help make inference more efficient on edge devices or web environments. Finally, explore how adaptive scheduling strategies between CPU and GPU can be used to balance latency and throughput for real-time web inference systems, especially when handling dynamic workloads or multiple simultaneous user sessions. Provide a comparative perspective on why certain tasks might still perform better on CPU versus GPU, and how intelligent co-scheduling could maximize resource utilization without degrading user experience."
]

# ✅ Test modes
MODES = ["cpu", "gpu", "adaptive"]

for mode in MODES:
    print(f"\n⚙️ Running tests in {mode.upper()} mode...\n")
    for prompt in TEST_PROMPTS:
        payload = {"prompt": prompt, "mode": mode}
        start = time.time()
        try:
            response = requests.post(API_URL, json=payload)
            latency = time.time() - start
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Prompt len={len(prompt.split())} | Model={data['selected_model']} | Latency={round(data['latency_s'],2)}s | Mode={mode}")
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"⚠️ Request failed: {e}")
        time.sleep(2)  # avoid server overload

print("\n🎯 All test runs complete! Check logs.csv for results.\n")
