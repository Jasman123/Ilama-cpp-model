# streaming_intermediate.py
import requests
import json
import time
import sys

def stream_generate(prompt: str, max_tokens: int = 300) -> dict:
    """
    Stream tokens from llama-server, display them live, and return stats.
    Returns a dict with: full_text, tokens_generated, elapsed_seconds, tokens_per_second
    """
    
    start_time = time.time()      # Record when we started
    full_text = []                 # Collect all tokens to return later
    token_count = 0                # Track how many tokens we got
    
    # Make the streaming request
    response = requests.post(
        "http://localhost:8080/completion",
        json={
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.7,
            "stream": True,
        },
        stream=True,
        timeout=120
    )
    
    print("\n--- Response ---")
    
    for line in response.iter_lines():
        if not line:
            continue  # Skip blank separator lines
            
        line_text = line.decode("utf-8")
        
        if not line_text.startswith("data: "):
            continue  # Skip non-data lines (SSE comments start with ":")
        
        # Parse the JSON chunk
        chunk = json.loads(line_text[6:])
        token = chunk["content"]
        
        # Append to our collector and display
        full_text.append(token)
        print(token, end="", flush=True)  # flush=True = show immediately
        token_count += 1
        
        if chunk.get("stop", False):
            break  # Generation complete
    
    elapsed = time.time() - start_time
    
    return {
        "full_text": "".join(full_text),
        "tokens_generated": token_count,
        "elapsed_seconds": round(elapsed, 2),
        "tokens_per_second": round(token_count / elapsed, 1) if elapsed > 0 else 0
    }


if __name__ == "__main__":
    stats = stream_generate("Explain quantum entanglement simply:")
    
    print(f"\n\n--- Stats ---")
    print(f"Tokens: {stats['tokens_generated']}")
    print(f"Time: {stats['elapsed_seconds']}s")
    print(f"Speed: {stats['tokens_per_second']} tokens/sec")