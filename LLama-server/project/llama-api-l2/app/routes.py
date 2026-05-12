# app/routes.py — Level 2
#
# WHY APIRouter instead of putting routes in main.py:
#   An APIRouter is a "mini-app" — you define endpoints here and
#   attach them to the main FastAPI app in main.py.
#   This lets you split a huge app into logical groups.

import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

# Import our own modules
from app.models import (
    GenerateRequest, GenerateResponse,
    ChatRequest, HealthResponse,
)
from app.client import LlamaClient

# Create the router — think of it as a mini FastAPI app
router = APIRouter()

# Create ONE client instance for the whole module.
# We read the URL from environment so it's configurable.
_client = LlamaClient(
    base_url=os.getenv("LLAMA_SERVER_URL", "http://localhost:8080")
)


# ── Health ────────────────────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse)
def health():
    """Check both this API and llama-server."""
    return HealthResponse(
        api_status="ok",
        llama_server="ok" if _client.is_healthy() else "unreachable",
        llama_url=_client.base_url,
    )


# ── Generate ──────────────────────────────────────────────────────────────────
@router.post("/generate", response_model=GenerateResponse)
def generate(body: GenerateRequest):
    """
    Send a raw prompt, get text back.
    This is a direct /completion call — no chat format, just text continuation.
    """
    try:
        result = _client.complete(
            prompt=body.prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            stop=body.stop,
        )
        return GenerateResponse(
            text=result["content"],
            tokens_used=result.get("tokens_predicted", 0),
            prompt_tokens=result.get("tokens_evaluated", 0),
        )
    except Exception as e:
        # HTTPException tells FastAPI to return a proper HTTP error response
        # status_code=502 means "Bad Gateway" — our upstream (llama-server) failed
        raise HTTPException(status_code=502, detail=str(e))


# ── Chat ──────────────────────────────────────────────────────────────────────
@router.post("/chat")
def chat(body: ChatRequest):
    """
    Multi-turn conversation endpoint.
    Accepts a full message history; returns the assistant's reply.

    If stream=True, returns Server-Sent Events (tokens arrive one by one).
    If stream=False, returns a plain JSON response.
    """

    # Convert our Pydantic models to plain dicts for the client
    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    if body.stream:
        # STREAMING PATH
        # We need to build the prompt from messages ourselves because
        # LlamaClient.stream_complete takes a raw string prompt.
        # A simple approach: join messages into a formatted string.
        prompt = _build_prompt_from_messages(messages)

        def token_generator():
            """
            Inner generator function that yields SSE-formatted strings.
            SSE format: "data: <text>\n\n"
            The client reads these and extracts the content after "data: "
            """
            for token in _client.stream_complete(
                prompt=prompt,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
            ):
                # SSE format: each event is "data: <content>\n\n"
                yield f"data: {token}\n\n"
            # Signal end of stream
            yield "data: [DONE]\n\n"

        # StreamingResponse sends data as it's generated
        # media_type tells the browser this is SSE
        return StreamingResponse(
            token_generator(),
            media_type="text/event-stream",
        )

    else:
        # NON-STREAMING PATH
        try:
            result = _client.chat_complete(
                messages=messages,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
            )
            # OpenAI-compatible response format
            return result
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))


def _build_prompt_from_messages(messages: list) -> str:
    """
    Convert a list of chat messages into a single prompt string.
    This is a simple format — real models have their own chat templates.
    """
    parts = []
    for msg in messages:
        role = msg["role"].capitalize()
        parts.append(f"{role}: {msg['content']}")
    parts.append("Assistant:")  # The model should continue from here
    return "\n".join(parts)