# llama_client.py
# Purpose: A reusable, well-structured client for llama-server
# This is the module you'll import in every other file

import requests
import json
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GenerationConfig:
    """
    All the settings for a single generation request.
    Using a dataclass means we get auto-generated __init__ and type hints.
    """
    prompt: str                           # Required: your input text
    max_tokens: int = 512                 # How many tokens to generate
    temperature: float = 0.7              # Randomness level
    top_p: float = 0.9                    # Nucleus sampling cutoff
    top_k: int = 40                       # Top-K candidates
    stop: list = field(default_factory=lambda: ["\n\n"])  # Stop sequences
    seed: int = -1                        # -1 = random, any other = reproducible
    stream: bool = False                  # Enable streaming (covered in M3)


class LlamaServerClient:
    """
    A client for the llama-server HTTP API.
    Wraps all HTTP logic so callers just think about prompts and responses.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        # Build the base URL once, use it everywhere
        self.base_url = f"http://{host}:{port}"
        # Create a session — reuses TCP connections for better performance
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def health_check(self) -> bool:
        """Returns True if the server is up and ready."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            # status_code 200 means HTTP "OK"
            return response.status_code == 200 and response.json().get("status") == "ok"
        except requests.RequestException:
            # If anything goes wrong (connection refused, timeout), return False
            return False
    
    def count_tokens(self, text: str) -> int:
        """Ask the server how many tokens a string would use."""
        payload = {"content": text}
        response = self.session.post(f"{self.base_url}/tokenize", json=payload)
        result = response.json()
        # The server returns a list of token IDs; the count is its length
        return len(result.get("tokens", []))
    
    def complete(self, config: GenerationConfig) -> dict:
        """
        Send a completion request and return the full response dict.
        Includes content, tokens_predicted, timing info, etc.
        """
        # Build the request payload from our config object
        payload = {
            "prompt": config.prompt,
            "n_predict": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "stop": config.stop,
            "seed": config.seed,
            "stream": config.stream,
        }
        
        response = self.session.post(
            f"{self.base_url}/completion",
            json=payload,
            timeout=120  # 2 minutes for long generations
        )
        
        # Raise an exception if the server returned an error code (4xx, 5xx)
        response.raise_for_status()
        
        return response.json()
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Convenience method: send a prompt, get back just the text.
        **kwargs lets you override any GenerationConfig field:
        e.g. generate("Hello", temperature=0.5, max_tokens=100)
        """
        config = GenerationConfig(prompt=prompt, **kwargs)
        result = self.complete(config)
        return result["content"]


# --- Sample usage (runs only when you execute this file directly) ---
if __name__ == "__main__":
    # Create a client pointing at our local server
    client = LlamaServerClient(host="localhost", port=8080)
    
    # Check the server is running before we try to use it
    if not client.health_check():
        print("Server is not running! Start it first.")
        exit(1)
    
    # Count tokens in a string (useful for context window management)
    token_count = client.count_tokens("How many tokens is this sentence?")
    print(f"Token count: {token_count}")
    
    # Simple generation
    response = client.generate(
        "What are the three laws of thermodynamics? Answer briefly:",
        temperature=0.3,    # Lower = more factual
        max_tokens=150
    )
    print("\nResponse:", response)