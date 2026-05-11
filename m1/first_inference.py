# first_inference.py
# Milestone 1 — Your very first local LLM inference
# Goal: load a GGUF model onto your RTX 3050 and get a real text response

from llama_cpp import Llama  # the main class — wraps the C++ engine in Python

# ─────────────────────────────────────────
# STEP 1 — Load the model
# ─────────────────────────────────────────
# Think of Llama() like opening a very intelligent file
# It reads the GGUF, loads weights into VRAM, and prepares the engine

print("Loading model... (this takes 5-15 seconds)")

llm = Llama(
    model_path="./models/qwen3-1.7b.gguf",  # path to your GGUF model file
    n_gpu_layers=-1,  # -1 = move ALL layers to GPU — maximum speed
                      #  0 = CPU only (slow)
                      # 10 = partial GPU (if VRAM is tight)
    n_ctx=2048,       # context window: how many tokens the model can "see"
                      # 2048 = ~1500 words of memory
    verbose=True,     # show loading details — lets us confirm GPU is used
)

print("\n" + "="*50)
print("MODEL LOADED SUCCESSFULLY")
print("="*50 + "\n")

# ─────────────────────────────────────────
# STEP 2 — Run inference (ask the model something)
# ─────────────────────────────────────────
# llm() sends your prompt through the model and returns a response
# Everything happens locally — no internet, no API key, no cloud

response = llm(
    "Q: What is the capital of Indonesia? A:",  # your prompt
    max_tokens=64,    # stop after 64 tokens maximum
    temperature=0.7,  # 0.0 = deterministic, 1.0 = very creative
                      # 0.7 = good balance for factual questions
    echo=False,       # False = don't repeat the prompt in output
    stop=["Q:"],      # stop generating if model starts a new question
)

# ─────────────────────────────────────────
# STEP 3 — Read the response
# ─────────────────────────────────────────
# The response is a Python dictionary that looks like this:
# {
#   "choices": [{"text": "Jakarta", "finish_reason": "stop"}],
#   "usage":   {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
# }

generated_text = response["choices"][0]["text"]  # extract just the text
finish_reason  = response["choices"][0]["finish_reason"]  # why it stopped

print("PROMPT:")
print("  Q: What is the capital of Indonesia? A:")
print("\nRESPONSE:")
print(f"  {generated_text.strip()}")
print(f"\nFinished because: {finish_reason}")

print("\n" + "="*50)
print("TOKEN USAGE:")
print(f"  Prompt tokens     : {response['usage']['prompt_tokens']}")
print(f"  Completion tokens : {response['usage']['completion_tokens']}")
print(f"  Total tokens      : {response['usage']['total_tokens']}")
print("="*50)