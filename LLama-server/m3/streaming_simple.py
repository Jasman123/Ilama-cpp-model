# streaming_simple.py
import requests
import json

# Send a streaming request and print each token as it arrives
response = requests.post(
    "http://localhost:8080/completion",
    json={
        "prompt": "Tell me a short story:",
        "n_predict": 200,
        "stream": True,   # This is the key flag — tells server to stream
    },
    stream=True           # This tells requests to not buffer the response
)

# iter_lines() reads the response body one line at a time as it arrives
for line in response.iter_lines():
    if line:  # Skip empty lines (SSE uses blank lines as separators)
        # Each line looks like: b"data: {...json...}"
        # We need to strip the "data: " prefix first
        line_text = line.decode("utf-8")   # Convert bytes to string
        
        if line_text.startswith("data: "):
            # Remove the "data: " prefix (6 characters)
            json_str = line_text[6:]
            
            # Parse the JSON payload
            chunk = json.loads(json_str)
            
            # Print the token without a newline, flush immediately so it shows
            print(chunk["content"], end="", flush=True)
            
            # stop=True means the model finished generating
            if chunk.get("stop", False):
                break

print()  # Final newline after all tokens are done