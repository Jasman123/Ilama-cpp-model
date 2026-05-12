# model_manager.py
# Milestone 2 — A reusable ModelManager class
#
# What this solves:
#   - Load model ONCE, reuse for every request
#   - Correct Qwen3 chat format (ChatML)
#   - Clean control over all generation parameters
#   - Streaming output token by token
#   - Safe error handling

import time                          # for measuring load/generation speed
from llama_cpp import Llama          # the core C++ engine wrapper


class ModelManager:
    """
    A wrapper around llama_cpp.Llama that:
    - Loads the model once and keeps it in VRAM
    - Formats prompts correctly for Qwen3 (ChatML format)
    - Exposes clean generate() and stream() methods
    - Tracks basic performance stats
    """

    def __init__(self, model_path: str, n_gpu_layers: int = -1, n_ctx: int = 2048):
        """
        __init__ runs when you do: manager = ModelManager(...)
        It stores the configuration but does NOT load the model yet.
        Loading is deferred to .load() so you control exactly when it happens.

        Args:
            model_path   : path to the .gguf file
            n_gpu_layers : how many layers to put on GPU (-1 = all)
            n_ctx        : context window size in tokens
        """
        self.model_path   = model_path    # store path for later
        self.n_gpu_layers = n_gpu_layers  # store GPU config
        self.n_ctx        = n_ctx         # store context size
        self.llm          = None          # model not loaded yet
        self.load_time    = None          # will store how long loading took

    # ──────────────────────────────────────────
    # LOADING
    # ──────────────────────────────────────────

    def load(self) -> None:
        """
        Load the model into VRAM.
        Call this ONCE when your application starts.
        After this, self.llm is ready to use.
        """
        print(f"Loading model from: {self.model_path}")
        print(f"GPU layers: {self.n_gpu_layers} | Context: {self.n_ctx} tokens")

        start = time.time()  # start timer

        self.llm = Llama(
            model_path    = self.model_path,
            n_gpu_layers  = self.n_gpu_layers,  # -1 = all layers on GPU
            n_ctx         = self.n_ctx,          # how much text model can see
            verbose       = False,               # silence the loading spam
        )

        self.load_time = time.time() - start  # measure how long it took
        print(f"Model loaded in {self.load_time:.2f}s ✅")

    def is_loaded(self) -> bool:
        """Check if model is ready — useful before trying to generate."""
        return self.llm is not None

    # ──────────────────────────────────────────
    # GENERATION
    # ──────────────────────────────────────────

    def generate(
        self,
        prompt      : str,
        system      : str   = "You are a helpful assistant.",
        max_tokens  : int   = 512,
        temperature : float = 0.7,
        top_p       : float = 0.95,
        top_k       : int   = 40,
        stop        : list  = None,
    ) -> dict:
        """
        Generate a response to a prompt.
        Returns a dictionary with the response text and stats.

        Args:
            prompt      : the user's message
            system      : instruction that sets the model's behavior
            max_tokens  : maximum tokens to generate
            temperature : creativity (0.0=focused, 1.0=creative)
            top_p       : nucleus sampling — only consider top P% of tokens
            top_k       : only consider top K token choices at each step
            stop        : list of strings that stop generation early
        """
        # Guard: don't try to generate if model isn't loaded
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call .load() first.")

        # Build the messages list — this is ChatML format
        # system = sets the personality/behavior of the assistant
        # user   = the actual question/request
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ]

        start = time.time()  # start timing generation

        # create_chat_completion() handles ChatML formatting automatically
        # It wraps messages in <|im_start|> tokens behind the scenes
        response = self.llm.create_chat_completion(
            messages    = messages,
            max_tokens  = max_tokens,
            temperature = temperature,
            top_p       = top_p,
            top_k       = top_k,
            stop        = stop or [],
        )

        elapsed = time.time() - start  # how long generation took

        # Extract the generated text from the response structure
        # response["choices"][0]["message"]["content"] = the actual text
        output_text   = response["choices"][0]["message"]["content"]
        finish_reason = response["choices"][0]["finish_reason"]
        usage         = response["usage"]

        # Calculate tokens per second
        tokens_generated = usage["completion_tokens"]
        tokens_per_sec   = tokens_generated / elapsed if elapsed > 0 else 0

        # Return a clean dictionary — easy to use in FastAPI later
        return {
            "text"          : output_text,
            "finish_reason" : finish_reason,
            "usage"         : usage,
            "performance"   : {
                "elapsed_sec"   : round(elapsed, 2),
                "tokens_per_sec": round(tokens_per_sec, 1),
            }
        }

    # ──────────────────────────────────────────
    # STREAMING
    # ──────────────────────────────────────────

    def stream(
        self,
        prompt      : str,
        system      : str   = "You are a helpful assistant.",
        max_tokens  : int   = 512,
        temperature : float = 0.7,
        top_p       : float = 0.95,
        top_k       : int   = 40,
    ):
        """
        Stream response tokens one by one as they are generated.
        This is a Python generator — use it with a for loop.

        Why streaming matters:
            Without streaming: user waits 10 seconds → sees full response
            With streaming:    user sees first word in 0.2 seconds → reads as it generates

        Usage:
            for token in manager.stream("hello"):
                print(token, end="", flush=True)
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call .load() first.")

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ]

        # stream=True tells llama.cpp to yield tokens as they are generated
        # instead of waiting for the full response
        response_stream = self.llm.create_chat_completion(
            messages    = messages,
            max_tokens  = max_tokens,
            temperature = temperature,
            top_p       = top_p,
            top_k       = top_k,
            stream      = True,   # ← this is the key difference
        )

        # Each chunk from the stream contains one or more tokens
        # We extract just the text content from each chunk
        for chunk in response_stream:
            delta = chunk["choices"][0]["delta"]         # what changed this step
            token = delta.get("content", "")             # extract text, default ""
            if token:                                    # skip empty chunks
                yield token                              # yield = send one token out

    # ──────────────────────────────────────────
    # INFO
    # ──────────────────────────────────────────

    def info(self) -> dict:
        """Return basic info about the loaded model."""
        return {
            "model_path"  : self.model_path,
            "n_gpu_layers": self.n_gpu_layers,
            "n_ctx"       : self.n_ctx,
            "load_time"   : f"{self.load_time:.2f}s" if self.load_time else "not loaded",
            "is_loaded"   : self.is_loaded(),
        }