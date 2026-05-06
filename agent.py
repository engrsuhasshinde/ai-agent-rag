from openai import OpenAI

client = OpenAI(
    base_url = "http://localhost:11434/v1",
    api_key = "ollama"
)

conversation = []

print("Suhas's Coding Assistant (🤖): Hello! (type 'exit' to quit)")
print("=" * 50)

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("Suhas's Coding Assistant (🤖): Goodbye!")
        break

    conversation.append({ "role": "user", "content": user_input })

    try:
        response = client.chat.completions.create(
            model = 'phi4-mini',
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert coding assistant."
                    )
                }
            ] + conversation
        )

        reply = response.choices[0].message.content
        conversation.append({ "role": "assistant", "content": reply })
        print(f"\n🤖: {reply}\n")

    except Exception as e:
        print(f"Error: {e}")
        conversation.pop()  # Remove the last unanswered user message from history if there's an error