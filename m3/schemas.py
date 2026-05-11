# schemas.py
# Milestone 3 — Request and Response shapes for our API
#
# Pydantic models do three things:
#   1. Define what fields are required/optional
#   2. Validate types automatically (string vs int vs float)
#   3. Generate API documentation automatically

from pydantic import BaseModel, Field
from typing import Optional


# ─────────────────────────────────────────
# REQUEST SCHEMAS — what clients send TO us
# ─────────────────────────────────────────

class GenerateRequest(BaseModel):
    """
    Shape of a POST /generate request body.
    Client sends JSON like:
    {
        "prompt": "What is Jakarta?",
        "max_tokens": 200,
        "temperature": 0.7
    }
    """
    # Field() lets us set defaults AND document each field
    prompt: str = Field(
        ...,                              # ... means REQUIRED — no default
        description="The user's input prompt",
        example="What is the capital of Indonesia?"
    )
    system: str = Field(
        default="You are a helpful assistant.",
        description="System prompt that sets the model's behavior"
    )
    max_tokens: int = Field(
        default=512,
        ge=1,          # ge = greater than or equal to 1
        le=4096,       # le = less than or equal to 4096
        description="Maximum number of tokens to generate"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,        # can't be negative
        le=2.0,        # cap at 2.0
        description="Creativity: 0=focused, 1=creative, 2=chaotic"
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold"
    )
    top_k: int = Field(
        default=40,
        ge=1,
        le=100,
        description="Top-K sampling: consider only top K tokens"
    )
    stream: bool = Field(
        default=False,
        description="If True, stream tokens as Server-Sent Events"
    )


class ChatMessage(BaseModel):
    """A single message in a conversation."""
    role: str = Field(
        ...,
        description="Who sent this message: 'user' or 'assistant' or 'system'",
        example="user"
    )
    content: str = Field(
        ...,
        description="The text content of the message",
        example="Hello, how are you?"
    )


class ChatRequest(BaseModel):
    """
    Shape of a POST /chat request body.
    Supports multi-turn conversation history.
    Client sends:
    {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What is Python?"}
        ]
    }
    """
    messages: list[ChatMessage] = Field(
        ...,
        description="Full conversation history as a list of messages"
    )
    max_tokens: int   = Field(default=512,  ge=1,   le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float       = Field(default=0.95,ge=0.0, le=1.0)
    top_k: int         = Field(default=40,  ge=1,   le=100)


# ─────────────────────────────────────────
# RESPONSE SCHEMAS — what we send BACK
# ─────────────────────────────────────────

class PerformanceInfo(BaseModel):
    """Speed metrics included in every response."""
    elapsed_sec    : float  # how long generation took
    tokens_per_sec : float  # generation speed


class UsageInfo(BaseModel):
    """Token usage included in every response."""
    prompt_tokens     : int
    completion_tokens : int
    total_tokens      : int


class GenerateResponse(BaseModel):
    """
    Shape of the response from /generate and /chat.
    Client receives:
    {
        "text": "Jakarta is the capital of Indonesia.",
        "finish_reason": "stop",
        "usage": {...},
        "performance": {...}
    }
    """
    text          : str
    finish_reason : str
    thinking      : Optional[str] = None  # Qwen3 reasoning (if present)
    usage         : UsageInfo
    performance   : PerformanceInfo


class HealthResponse(BaseModel):
    """Shape of the response from /health."""
    status      : str   # "ok" or "error"
    model_loaded: bool
    model_path  : str
    load_time   : str
    gpu_layers  : int
    context_size: int