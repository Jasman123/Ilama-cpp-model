# streaming_client.py
# Purpose: Production-grade streaming client, handles edge cases properly

import requests
import json
import time
from typing import Generator, Iterator


class StreamingClient:
    """
    Handles SSE streaming from llama-server.
    Designed to be used as a context manager or standalone.
    """
    
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url
        self.session = requests.Session()
    
    def stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list = None
    ) -> Generator[str, None, None]:
        """
        A Python generator that yields one token at a time.
        
        A generator is a function that 'yields' values one by one instead
        of returning all at once — perfect for streaming token by token.
        
        Usage:
            for token in client.stream("Hello"):
                print(token, end="", flush=True)
        """
        
        if stop is None:
            stop = ["\n\n"]
        
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": stop,
            "stream": True,
        }
        
        try:
            # stream=True keeps the connection open for chunked reading
            with self.session.post(
                f"{self.server_url}/completion",
                json=payload,
                stream=True,
                timeout=120
            ) as response:
                
                # raise_for_status() throws an exception for 4xx/5xx errors
                response.raise_for_status()
                
                for raw_line in response.iter_lines():
                    if not raw_line:
                        continue
                    
                    line = raw_line.decode("utf-8", errors="replace")
                    
                    if not line.startswith("data: "):
                        continue
                    
                    json_str = line[6:].strip()
                    
                    # Guard: skip malformed chunks rather than crash
                    if not json_str or json_str == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(json_str)
                    except json.JSONDecodeError:
                        continue  # Skip unparseable chunks
                    
                    token = chunk.get("content", "")
                    
                    if token:
                        yield token  # Hand this token to the caller
                    
                    if chunk.get("stop", False):
                        return  # Generator is done
                        
        except requests.RequestException as e:
            # Yield an error marker so callers know something went wrong
            yield f"\n[Stream error: {e}]"
    
    def stream_to_string(self, prompt: str, **kwargs) -> str:
        """Convenience: collect all streamed tokens into a single string."""
        return "".join(self.stream(prompt, **kwargs))
    
    def stream_with_display(self, prompt: str, **kwargs) -> str:
        """Stream tokens, display them live, and return the full text."""
        tokens = []
        for token in self.stream(prompt, **kwargs):
            print(token, end="", flush=True)
            tokens.append(token)
        print()  # Final newline
        return "".join(tokens)


# --- Sample usage ---
if __name__ == "__main__":
    client = StreamingClient()
    
    print("Streaming response:")
    full = client.stream_with_display(
        "What is the meaning of exercise?",
        max_tokens=200,
        temperature=0.5
    )
    print(f"\nTotal characters: {len(full)}")