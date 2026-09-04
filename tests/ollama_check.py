import ollama

response = ollama.chat(
    model="llama3.1",
    messages=[
        {
            "role": "user",
            "content": "Say exactly: Ollama is working locally."
        }
    ],
)

print(response.message.content)