# checkpoint.py — run this before every milestone
# Save it as checkpoint.py and run: python checkpoint.py

import sys

print("=" * 50)
print("ENVIRONMENT CHECKPOINT")
print("=" * 50)

# --- Python version ---
print(f"\n[Python]  {sys.version}")

# --- CUDA / GPU via torch ---
try:
    import torch  # torch is used here only as a GPU probe
    cuda_ok = torch.cuda.is_available()
    print(f"[CUDA]    available = {cuda_ok}")
    if cuda_ok:
        print(f"[GPU]     {torch.cuda.get_device_name(0)}")
        print(f"[VRAM]    {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    else:
        print("[GPU]     none detected — check CUDA toolkit install")
except ImportError:
    print("[torch]   NOT installed — install with: pip install torch")

# --- llama-cpp-python ---
try:
    import llama_cpp
    print(f"[llama-cpp-python]  {llama_cpp.__version__}")
except ImportError:
    print("[llama-cpp-python]  NOT installed")

# --- FastAPI ---
try:
    import fastapi
    print(f"[FastAPI]  {fastapi.__version__}")
except ImportError:
    print("[FastAPI]  NOT installed")

# --- Pydantic ---
try:
    import pydantic
    print(f"[Pydantic] {pydantic.__version__}")
except ImportError:
    print("[Pydantic] NOT installed")

# --- Uvicorn ---
try:
    import uvicorn
    print(f"[Uvicorn]  {uvicorn.__version__}")
except ImportError:
    print("[Uvicorn]  NOT installed")

print("\n" + "=" * 50)
print("Checkpoint complete.")
print("=" * 50)