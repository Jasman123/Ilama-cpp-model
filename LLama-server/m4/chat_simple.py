# chat_simple.py
import requests

def chat(messages: list) -> str:
    """Send a conversation and get the assistant's reply."""
    
    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "local",   # llama-server ignores this, but it's required
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.7,
        }
    )
    
    # OpenAI format: choices[0].message.content
    return response.json()["choices"][0]["message"]["content"]


# Build a multi-turn conversation
conversation = [
    # System message sets the AI's persona and rules
    {"role": "system", "content": "You are a helpful assistant that explains things simply."},
    # First user message
    {"role": "user", "content": "What is Python?"},
]

# First turn
reply1 = chat(conversation)
print("Assistant:", reply1)

# Add the reply to history so the model "remembers" it
conversation.append({"role": "assistant", "content": reply1})

# Second turn — model now knows the context of the first exchange
conversation.append({"role": "user", "content": "Can you give me a simple example?"})
reply2 = chat(conversation)
print("Assistant:", reply2)