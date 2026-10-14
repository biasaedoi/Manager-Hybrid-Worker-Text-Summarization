import ollama

MODEL = "qwen3.5:0.8b"

response = ollama.chat(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": "Summarize this sentence: The cat is sitting on the table."
        }
    ]
)

print("TYPE:", type(response))
print("RESPONSE:")
print(response)