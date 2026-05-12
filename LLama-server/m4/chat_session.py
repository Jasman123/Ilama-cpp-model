# chat_session.py
import requests

class ChatSession:
    """
    Manages a multi-turn conversation automatically.
    You just call .chat(), it handles adding messages to history.
    """
    
    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        # history stores the full conversation so far
        # We always start with the system message
        self.history = [{"role": "system", "content": system_prompt}]
        self.server_url = "http://localhost:8080"
    
    def chat(self, user_message: str, stream: bool = False) -> str:
        """Add a user message, get assistant reply, and update history."""
        
        # Step 1: Add the user's message to history
        self.history.append({"role": "user", "content": user_message})
        
        # Step 2: Send the full history to the server
        response = requests.post(
            f"{self.server_url}/v1/chat/completions",
            json={
                "model": "local",
                "messages": self.history,   # Full history = model has memory
                "max_tokens": 500,
                "temperature": 0.7,
                "stream": stream,
            },
            stream=stream,
            timeout=120
        )
        
        if not stream:
            # Non-streaming: get full reply at once
            reply = response.json()["choices"][0]["message"]["content"]
        else:
            # Streaming: collect tokens as they arrive
            reply = self._collect_stream(response)
        
        # Step 3: Add the assistant's reply to history
        # This is what makes it "remember" previous turns
        self.history.append({"role": "assistant", "content": reply})
        
        return reply
    
    def _collect_stream(self, response) -> str:
        """Read a streaming chat response and return the full text."""
        import json
        
        full_reply = []
        
        for line in response.iter_lines():
            if not line:
                continue
            
            text = line.decode("utf-8")
            
            if not text.startswith("data: "):
                continue
            
            json_str = text[6:]
            
            if json_str == "[DONE]":
                break
            
            try:
                chunk = json.loads(json_str)
                # Chat completions use choices[0].delta.content for streaming
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    print(content, end="", flush=True)
                    full_reply.append(content)
                    
                # finish_reason = "stop" means generation is done
                if chunk["choices"][0].get("finish_reason") == "stop":
                    break
            except (json.JSONDecodeError, KeyError):
                continue
        
        print()  # Final newline after streaming
        return "".join(full_reply)
    
    def clear(self):
        """Reset the conversation (keep only the system message)."""
        system = self.history[0]
        self.history = [system]
    
    def context_size(self) -> int:
        """Rough estimate of how many characters are in the history."""
        return sum(len(m["content"]) for m in self.history)


if __name__ == "__main__":
    session = ChatSession(
        system_prompt="You are a Python tutor. Explain concepts clearly with short examples."
    )
    
    # Multi-turn conversation
    questions = [
        "What is a list comprehension?",
        "Can you show me an example with filtering?",
        "How is that different from a regular for loop?",
    ]
    
    for question in questions:
        print(f"\nYou: {question}")
        reply = session.chat(question)
        print(f"Assistant: {reply}")
        print(f"(History size: {len(session.history)} messages)")