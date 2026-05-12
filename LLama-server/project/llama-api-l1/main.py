# main.py — Level 1: Beginner FastAPI + llama-server
#
# WHAT WE ARE BUILDING:
#   A tiny web service with two endpoints:
#     GET  /health   → tells the caller if everything is alive
#     POST /generate → takes a prompt, asks llama-server, returns the answer
#
# WHY THIS STRUCTURE:
#   Everything in one file so you can read top-to-bottom and understand
#   the whole app without jumping between files.

import os          # os lets us read environment variables (settings)
import requests    # requests is for making HTTP calls to llama-server
from fastapi import FastAPI           # FastAPI is the web framework
from fastapi.responses import JSONResponse  # JSONResponse wraps dicts as HTTP JSON
from pydantic import BaseModel        # BaseModel validates incoming JSON
from dotenv import load_dotenv        # load_dotenv reads our .env file

# ── 1. Load configuration ──────────────────────────────────────────────────────
# load_dotenv() reads .env and puts those values into the environment
# so os.getenv() can find them below
load_dotenv()

# Read the llama-server URL from .env; fall back to localhost:8080 if not set
LLAMA_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080")

# ── 2. Create the FastAPI app ──────────────────────────────────────────────────
# FastAPI() is the application object — think of it as the "brain"
# that receives requests and routes them to the right function
app = FastAPI(
    title="Llama API — Level 1",   # Shows up in /docs
    description="Beginner: one file, two endpoints",
    version="1.0.0",
)

# ── 3. Define request/response shapes with Pydantic ──────────────────────────
# BaseModel lets us describe what JSON we expect.
# If the caller sends something wrong, FastAPI rejects it automatically.
class GenerateRequest(BaseModel):
    prompt: str           # Required: the text to send to the model
    max_tokens: int = 200 # Optional: how many tokens to generate (default 200)
    temperature: float = 0.7  # Optional: 0=predictable, 1=creative


# ── 4. Health endpoint ────────────────────────────────────────────────────────
# @app.get("/health") means:
#   "When someone sends a GET request to /health, call this function"
@app.get("/health")
def health():
    """
    Check if our API is alive AND if llama-server is reachable.
    Returns a JSON dict with status information.
    """
    # First check our own status
    our_status = "ok"

    # Then try to reach llama-server's health endpoint
    try:
        # requests.get sends an HTTP GET and waits up to 3 seconds
        response = requests.get(f"{LLAMA_URL}/health", timeout=3)
        # status_code 200 means "HTTP OK" — the server replied normally
        llama_status = "ok" if response.status_code == 200 else "error"
    except requests.RequestException:
        # RequestException covers: connection refused, timeout, DNS failure
        llama_status = "unreachable"

    # Return a plain dict — FastAPI automatically converts it to JSON
    return {
        "api_status": our_status,
        "llama_server": llama_status,
        "llama_url": LLAMA_URL,
    }


# ── 5. Generate endpoint ──────────────────────────────────────────────────────
# @app.post("/generate") means:
#   "When someone sends a POST request to /generate, call this function"
# The parameter `body: GenerateRequest` tells FastAPI:
#   "Parse the request JSON body into a GenerateRequest object"
@app.post("/generate")
def generate(body: GenerateRequest):
    """
    Send a prompt to llama-server and return the generated text.

    Request body (JSON):
        {"prompt": "What is Python?", "max_tokens": 200, "temperature": 0.7}

    Response (JSON):
        {"text": "Python is...", "tokens_used": 45, "model_url": "..."}
    """

    # Build the payload dict that llama-server /completion expects
    llama_payload = {
        "prompt": body.prompt,        # From the caller's request
        "n_predict": body.max_tokens, # llama-server calls this n_predict
        "temperature": body.temperature,
        "stop": ["\n\n"],             # Stop generating at double newline
        "stream": False,              # Level 1: no streaming yet
    }

    # Try to call llama-server; handle errors gracefully
    try:
        # requests.post sends our payload to llama-server
        # json=llama_payload automatically sets Content-Type: application/json
        response = requests.post(
            f"{LLAMA_URL}/completion",
            json=llama_payload,
            timeout=60,  # Give model up to 60 seconds to respond
        )

        # Raise a Python exception if the HTTP status is 4xx or 5xx
        response.raise_for_status()

        # Parse the JSON response body into a Python dict
        llama_data = response.json()

        # Return the generated text back to our caller
        return {
            "text": llama_data["content"],           # The generated text
            "tokens_used": llama_data.get("tokens_predicted", 0),
            "model_url": LLAMA_URL,
        }

    except requests.ConnectionError:
        # This happens when llama-server is not running
        # status_code=503 means "Service Unavailable"
        return JSONResponse(
            status_code=503,
            content={"error": "llama-server is not running", "url": LLAMA_URL},
        )
    except requests.Timeout:
        return JSONResponse(
            status_code=504,
            content={"error": "llama-server took too long to respond"},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


# ── 6. Run the server ─────────────────────────────────────────────────────────
# This block only runs when you execute: python main.py
# It starts uvicorn (the ASGI web server that runs FastAPI)
if __name__ == "__main__":
    import uvicorn
    # host="0.0.0.0" means "listen on all network interfaces"
    # port=8000  means our API lives at http://localhost:8000
    # reload=True means "restart automatically when you save main.py"
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)