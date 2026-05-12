# app/client.py — Level 2
#
# WHY: Putting all llama-server HTTP logic here means:
#   1. Routes never call requests.post directly — they call the client
#   2. If llama-server's API changes, you fix it in one place
#   3. You can test the client independently of FastAPI

import json
import requests
from typing import Generator, List, Dict


class LlamaClient:
    """
    Handles all communication with a running llama-server instance.
    Create one instance and share it across your whole app.
    """

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        # Session reuses the TCP connection → faster than creating a new
        # connection for every request
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def is_healthy(self) -> bool:
        """Returns True if llama-server responds to /health."""
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=3)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def complete(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: List[str] = None,
    ) -> dict:
        """
        Send a prompt to /completion and return the full response dict.
        Use this for non-streaming responses.
        """
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop or ["\n\n"],
            "stream": False,
        }

        response = self.session.post(
            f"{self.base_url}/completion",
            json=payload,
            timeout=90,
        )
        response.raise_for_status()
        return response.json()

    def stream_complete(
        self,
        prompt: str,
        max_tokens: int = 400,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        A generator that yields one token string at a time.
        The caller iterates over this to get tokens as they arrive.

        WHY a generator: generators are lazy — they produce one value,
        pause, and wait. Perfect for streaming because you don't need
        to buffer the whole response before showing it.
        """
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stream": True,  # Key flag: tells llama-server to stream
        }

        with self.session.post(
            f"{self.base_url}/completion",
            json=payload,
            stream=True,     # Tells requests: don't buffer, give us chunks
            timeout=90,
        ) as response:
            response.raise_for_status()

            for raw_line in response.iter_lines():
                if not raw_line:
                    continue  # SSE uses blank lines as separators

                line = raw_line.decode("utf-8", errors="replace")

                if not line.startswith("data: "):
                    continue

                json_str = line[6:].strip()

                if not json_str or json_str == "[DONE]":
                    return

                try:
                    chunk = json.loads(json_str)
                except json.JSONDecodeError:
                    continue

                token = chunk.get("content", "")
                if token:
                    yield token  # Hand this token to whoever is iterating

                if chunk.get("stop", False):
                    return

    def chat_complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 400,
        temperature: float = 0.7,
    ) -> dict:
        """
        Send a multi-turn conversation to /v1/chat/completions.
        messages = [{"role": "user", "content": "Hello"}, ...]
        """
        payload = {
            "model": "local",    # llama-server ignores this but it's required
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=90,
        )
        response.raise_for_status()
        return response.json()