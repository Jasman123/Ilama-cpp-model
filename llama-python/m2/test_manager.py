# test_manager.py
# Milestone 2 — Test all ModelManager features
# Run this to verify everything works before moving to FastAPI

from model_manager import ModelManager  # import our new class

MODEL_PATH = "./models/qwen3-1.7b.gguf"  # path to your model

# ─────────────────────────────────────────
# STEP 1 — Create and load the manager
# ─────────────────────────────────────────
# Creating the object is instant — no model loaded yet
manager = ModelManager(
    model_path   = MODEL_PATH,
    n_gpu_layers = -1,    # all layers on GPU
    n_ctx        = 2048,  # context window
)

# Load model into VRAM — this is the slow part (happens once)
manager.load()

# Print model info
print("\n📋 Model Info:")
for key, value in manager.info().items():
    print(f"   {key}: {value}")

# ─────────────────────────────────────────
# STEP 2 — Test basic generation
# ─────────────────────────────────────────
print("\n" + "="*50)
print("TEST 1 — Basic generation")
print("="*50)

result = manager.generate(
    prompt      = "What is the capital of Indonesia? Answer in one sentence.",
    max_tokens  = 100,
    temperature = 0.7,
)

print(f"Response : {result['text']}")
print(f"Speed    : {result['performance']['tokens_per_sec']} tokens/sec")
print(f"Tokens   : {result['usage']['completion_tokens']} generated")
print(f"Stopped  : {result['finish_reason']}")

# ─────────────────────────────────────────
# STEP 3 — Test with custom system prompt
# ─────────────────────────────────────────
print("\n" + "="*50)
print("TEST 2 — Custom system prompt")
print("="*50)

result = manager.generate(
    prompt     = "Explain what an API is.",
    system     = "You are a teacher explaining concepts to a beginner. "
                 "Use simple words and keep answers under 3 sentences.",
    max_tokens = 150,
    temperature= 0.7,
)

print(f"Response : {result['text']}")
print(f"Speed    : {result['performance']['tokens_per_sec']} tokens/sec")

# ─────────────────────────────────────────
# STEP 4 — Test temperature effect
# ─────────────────────────────────────────
print("\n" + "="*50)
print("TEST 3 — Temperature comparison")
print("="*50)

# Low temperature — focused, predictable
result_low = manager.generate(
    prompt      = "Complete this sentence: The sky is",
    max_tokens  = 20,
    temperature = 0.1,   # almost deterministic
)

# High temperature — creative, varied
result_high = manager.generate(
    prompt      = "Complete this sentence: The sky is",
    max_tokens  = 20,
    temperature = 1.2,   # very creative
)

print(f"Temp 0.1 (focused)  : {result_low['text'].strip()}")
print(f"Temp 1.2 (creative) : {result_high['text'].strip()}")

# ─────────────────────────────────────────
# STEP 5 — Test streaming
# ─────────────────────────────────────────
print("\n" + "="*50)
print("TEST 4 — Streaming output")
print("="*50)
print("Tokens appearing one by one:\n")

# The for loop receives one token at a time
# end="" means no newline between tokens
# flush=True means print immediately, don't buffer
for token in manager.stream(
    prompt      = "Count from 1 to 5 and say one fun fact about each number.",
    max_tokens  = 200,
    temperature = 0.7,
):
    print(token, end="", flush=True)  # print each token as it arrives

print("\n\n" + "="*50)
print("All tests complete ✅")
print("="*50)