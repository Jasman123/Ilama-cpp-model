# simple_client.py
# Purpose: Send a prompt to llama-server and print the response
# This is the simplest possible client

import requests  # requests is a Python library for making HTTP calls
import json      # json helps us work with JSON data

# The URL of our llama-server
# localhost means "this machine", 8080 is the port we started it on
SERVER_URL = "http://localhost:8080"

def generate(prompt: str, max_tokens: int = 200) -> str:
    """
    Send a prompt to llama-server and return the generated text.
    
    prompt: the text you want the model to continue or respond to
    max_tokens: how many tokens to generate (roughly 1 token ≈ 0.75 words)
    """
    
    # Build the request body as a Python dictionary
    # This will be converted to JSON and sent to the server
    payload = {
        "prompt": prompt,           # Your input text
        "n_predict": max_tokens,    # How many tokens to generate
        "temperature": 0.7,         # 0=deterministic, 1=normal, 2=wild
        "top_p": 0.9,               # Only consider top 90% probability tokens
        "top_k": 40,                # Only consider top 40 token candidates
        "stop": ["\n\n", "User:"],  # Stop generating when we see these strings
    }
    
    # Make the HTTP POST request
    # requests.post sends data to the server and waits for a response
    response = requests.post(
        f"{SERVER_URL}/completion",   # The endpoint URL
        json=payload,                  # json= converts dict to JSON automatically
        timeout=60                     # Give up if no response after 60 seconds
    )
    
    # response.json() parses the JSON response body into a Python dict
    result = response.json()
    
    # The actual generated text lives in the "content" key
    return result["content"]


# Run this only when we execute this file directly (not when imported)
if __name__ == "__main__":
    answer = generate("Explain what a neural network is in simple terms:")
    print(answer)