# test_api.py
# Milestone 3 — Test all API endpoints
# Run AFTER starting the server in a separate terminal

import requests   # pip install requests if missing
import json

BASE_URL = "http://localhost:8000"

def print_section(title: str):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

# ── TEST 1: Health check ──
print_section("TEST 1 — Health check")
response = requests.get(f"{BASE_URL}/health")
print(f"Status code : {response.status_code}")
print(f"Response    : {json.dumps(response.json(), indent=2)}")

# ── TEST 2: Generate ──
print_section("TEST 2 — Generate")
response = requests.post(f"{BASE_URL}/generate", json={
    "prompt"     : "What is the capital of Indonesia? One sentence only.",
    "max_tokens" : 100,
    "temperature": 0.7,
})
data = response.json()
print(f"Status  : {response.status_code}")
print(f"Answer  : {data['text']}")
print(f"Speed   : {data['performance']['tokens_per_sec']} tok/s")
if data.get("thinking"):
    print(f"Thinking: {data['thinking'][:80]}...")  # first 80 chars

# ── TEST 3: Chat (multi-turn) ──
print_section("TEST 3 — Multi-turn chat")
response = requests.post(f"{BASE_URL}/chat", json={
    "messages": [
        {"role": "user",      "content": "My name is Budi."},
        {"role": "assistant", "content": "Hello Budi! How can I help you?"},
        {"role": "user",      "content": "What is my name?"},
    ],
    "max_tokens": 50,
})
data = response.json()
print(f"Status  : {response.status_code}")
print(f"Answer  : {data['text']}")

# ── TEST 4: Streaming ──
print_section("TEST 4 — Streaming")
print("Tokens arriving in real time:\n")

with requests.post(
    f"{BASE_URL}/stream",
    json={"prompt": "Count to 3 and give one fact per number.", "max_tokens": 150},
    stream=True    # tell requests to not buffer the response
) as response:
    for line in response.iter_lines():       # read line by line
        if line:
            decoded = line.decode("utf-8")   # bytes → string
            if decoded.startswith("data: "): # SSE format
                token = decoded[6:]          # strip "data: " prefix
                if token == "[DONE]":
                    print("\n\n[Stream complete]")
                    break
                print(token, end="", flush=True)  # print without newline

print_section("All tests complete ✅")