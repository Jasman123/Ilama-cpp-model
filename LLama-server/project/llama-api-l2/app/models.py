# app/models.py — Level 2
#
# WHY: Pydantic models define the shape of data that flows through the API.
# Keeping them in one file means you can find and edit your data contracts
# in a single place, not scattered through route functions.

from pydantic import BaseModel, Field
from typing import Optional, List


# ── Request models (what callers send to us) ───────────────────────────────────

class GenerateRequest(BaseModel):
    """Body for POST /generate"""
    prompt: str = Field(..., min_length=1, description="Text to complete")
    # Field(...) means "required". Field(200) means "optional, default 200".
    max_tokens: int = Field(200, ge=1, le=4096)   # ge=1 means "at least 1"
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    stop: List[str] = Field(default_factory=lambda: ["\n\n"])


class ChatMessage(BaseModel):
    """A single message in a conversation."""
    # Literal type: only these exact string values are valid
    role: str = Field(..., description="'user', 'assistant', or 'system'")
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    """Body for POST /chat — a full conversation history."""
    messages: List[ChatMessage]
    max_tokens: int = Field(400, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    stream: bool = Field(False, description="Set true for token-by-token SSE")


# ── Response models (what we send back) ───────────────────────────────────────

class GenerateResponse(BaseModel):
    """Shape of the /generate response."""
    text: str
    tokens_used: int
    prompt_tokens: int = 0


class HealthResponse(BaseModel):
    """Shape of the /health response."""
    api_status: str
    llama_server: str
    llama_url: str
