# main.py
# Milestone 3 — FastAPI service wrapping our ModelManager
#
# Endpoints:
#   GET  /health     — is the server alive? is the model loaded?
#   POST /generate   — send a prompt, get a response
#   POST /chat       — send conversation history, get a response
#   POST /stream     — send a prompt, get tokens as they generate

import sys
import re
from pathlib import Path
from contextlib import asynccontextmanager

# Add parent directory to path so we can import model_manager from m2/
# In production (M6) we'll package this properly
sys.path.append(str(Path(__file__).parent.parent / "m2"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn

from model_manager import ModelManager   # our M2 class
from schemas import (                    # our request/response shapes
    GenerateRequest, GenerateResponse,
    ChatRequest,
    HealthResponse,
    UsageInfo, PerformanceInfo,
)

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
# In M6 we'll move this to a .env file
# For now, hardcode so everything is visible

MODEL_PATH   = "../models/qwen3-1.7b.gguf"  # relative to m3/
N_GPU_LAYERS = -1    # all layers on GPU
N_CTX        = 2048  # context window

# ─────────────────────────────────────────
# LIFESPAN — startup and shutdown events
# ─────────────────────────────────────────
# This is the modern FastAPI way to run code at startup/shutdown
# @asynccontextmanager means it's an async context manager
# Everything BEFORE yield runs at startup
# Everything AFTER yield runs at shutdown

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──
    print("Server starting — loading model...")
    app.state.manager = ModelManager(     # store manager on app.state
        model_path   = MODEL_PATH,        # accessible from any endpoint
        n_gpu_layers = N_GPU_LAYERS,
        n_ctx        = N_CTX,
    )
    app.state.manager.load()              # load into VRAM — happens ONCE
    print("Server ready ✅")

    yield  # server runs here — handling requests

    # ── SHUTDOWN ──
    print("Server shutting down...")
    # model is freed from VRAM automatically when manager is garbage collected


# ─────────────────────────────────────────
# APP CREATION
# ─────────────────────────────────────────

app = FastAPI(
    title       = "llama-cpp Model API",
    description = "Local LLM inference API powered by llama.cpp and Qwen3",
    version     = "0.1.0",
    lifespan    = lifespan,   # register our startup/shutdown handler
)


# ─────────────────────────────────────────
# HELPER — strip Qwen3 thinking tags
# ─────────────────────────────────────────

def parse_qwen3_response(text: str) -> tuple[str, str]:
    """
    Qwen3 wraps its reasoning in <think>...</think> tags.
    This splits the response into:
      - thinking : the internal reasoning (optional, for debugging)
      - answer   : the actual response to show the user

    Returns: (thinking_text, answer_text)
    """
    think_pattern = r"<think>(.*?)</think>"           # regex to find think block
    match = re.search(think_pattern, text, re.DOTALL) # DOTALL = match newlines too

    if match:
        thinking = match.group(1).strip()             # text inside <think>
        answer   = re.sub(think_pattern, "", text, flags=re.DOTALL).strip()
        return thinking, answer
    else:
        return "", text.strip()   # no thinking block — return as-is


# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    Call this to verify the server is running and the model is loaded.
    Returns model info and load status.
    """
    manager = app.state.manager
    info    = manager.info()

    return HealthResponse(
        status       = "ok" if manager.is_loaded() else "error",
        model_loaded = manager.is_loaded(),
        model_path   = info["model_path"],
        load_time    = info["load_time"],
        gpu_layers   = info["n_gpu_layers"],
        context_size = info["n_ctx"],
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate a response to a single prompt.

    Send:  {"prompt": "What is Jakarta?"}
    Get:   {"text": "Jakarta is the capital...", "usage": {...}}
    """
    manager = app.state.manager

    # HTTPException = FastAPI's way of returning error responses
    # status_code 503 = Service Unavailable
    if not manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Call our ModelManager — same as M2 but via HTTP now
        result = manager.generate(
            prompt      = request.prompt,
            system      = request.system,
            max_tokens  = request.max_tokens,
            temperature = request.temperature,
            top_p       = request.top_p,
            top_k       = request.top_k,
        )

        # Split out Qwen3 thinking from the actual answer
        thinking, answer = parse_qwen3_response(result["text"])

        return GenerateResponse(
            text          = answer,
            finish_reason = result["finish_reason"],
            thinking      = thinking if thinking else None,
            usage         = UsageInfo(**result["usage"]),
            performance   = PerformanceInfo(**result["performance"]),
        )

    except Exception as e:
        # status_code 500 = Internal Server Error
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=GenerateResponse)
async def chat(request: ChatRequest):
    """
    Multi-turn chat endpoint.
    Send the full conversation history, get the next response.

    Send:
    {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What is Python?"}
        ]
    }
    """
    manager = app.state.manager

    if not manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert our Pydantic ChatMessage objects to plain dicts
        # llama_cpp expects: [{"role": "user", "content": "..."}]
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]

        import time
        start = time.time()

        # Call llama_cpp directly for multi-turn (ModelManager.generate
        # only supports single-turn — we extend it here)
        response = manager.llm.create_chat_completion(
            messages    = messages,
            max_tokens  = request.max_tokens,
            temperature = request.temperature,
            top_p       = request.top_p,
            top_k       = request.top_k,
        )

        elapsed      = time.time() - start
        raw_text     = response["choices"][0]["message"]["content"]
        finish_reason= response["choices"][0]["finish_reason"]
        usage        = response["usage"]

        thinking, answer = parse_qwen3_response(raw_text)

        tokens_per_sec = usage["completion_tokens"] / elapsed if elapsed > 0 else 0

        return GenerateResponse(
            text          = answer,
            finish_reason = finish_reason,
            thinking      = thinking if thinking else None,
            usage         = UsageInfo(**usage),
            performance   = PerformanceInfo(
                elapsed_sec    = round(elapsed, 2),
                tokens_per_sec = round(tokens_per_sec, 1),
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream")
async def stream(request: GenerateRequest):
    """
    Streaming endpoint — returns tokens as Server-Sent Events (SSE).
    The client receives data in real time as the model generates.

    SSE format (what the client receives):
        data: Hello
        data:  world
        data: !
        data: [DONE]
    """
    manager = app.state.manager

    if not manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    def token_generator():
        """
        Inner generator function.
        Yields tokens in SSE format as the model generates them.
        """
        try:
            for token in manager.stream(
                prompt      = request.prompt,
                system      = request.system,
                max_tokens  = request.max_tokens,
                temperature = request.temperature,
                top_p       = request.top_p,
                top_k       = request.top_k,
            ):
                # SSE format: each event is "data: <content>\n\n"
                yield f"data: {token}\n\n"

            # Signal that streaming is complete
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    # StreamingResponse sends chunks as they arrive
    # media_type tells the client to expect Server-Sent Events
    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
    )


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",       # "filename:app_variable"
        host    = "0.0.0.0",  # listen on all interfaces
        port    = 8000,       # standard port
        reload  = False,      # True = auto-reload on code changes (dev only)
                              # False here because model reload is expensive
    )