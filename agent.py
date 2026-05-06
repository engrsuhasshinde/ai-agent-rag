from openai import OpenAI
import os
import json
from datetime import datetime
import re

# ------------------ CONFIG ------------------
SYSTEM_PROMPT = (
    "You are an expert coding assistant.\n"
    "Be concise and clear.\n"
    "If past conversation context is provided, USE it to answer.\n"
    "Do not ignore relevant past context."
)
CHAT_DIR = "chat_history"
MAX_HISTORY = 20

client = OpenAI(
    base_url = "http://localhost:11434/v1",
    api_key = "ollama"
)

# ------------------ FILE HANDLING ------------------
def create_chat_file():
    os.makedirs(CHAT_DIR, exist_ok = True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{CHAT_DIR}/chat_{timestamp}.json"

def save_conversation(filename, conversation):
    with open(filename, "w") as f:
        json.dump(conversation, f, indent = 4)

def load_all_chats():
    chats = []

    if not os.path.exists(CHAT_DIR):
        return chats

    for file in sorted(os.listdir(CHAT_DIR)):
        if file.endswith(".json"):
            try:
                with open(os.path.join(CHAT_DIR, file), "r") as f:
                    chats.extend(json.load(f))
            except Exception:
                continue

    return chats

def load_chats_by_date_range(start_date, end_date):
    chats = []

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    if not os.path.exists(CHAT_DIR):
        return chats

    for file in sorted(os.listdir(CHAT_DIR)):
        if not file.endswith(".json"):
            continue

        try:
            date_str = file.split("_")[1]
            file_date = datetime.strptime(date_str, "%Y-%m-%d")

            if start <= file_date <= end:
                with open(os.path.join(CHAT_DIR, file), "r") as f:
                    chats.extend(json.load(f))
        except Exception:
            continue

    return chats

# ------------------ MEMORY ------------------
def parse_command(user_input):
    pattern_all = r"^/load all:\s*(.*)"
    pattern_range = r"^/load range from (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2}):\s*(.*)"

    match_all = re.match(pattern_all, user_input, re.IGNORECASE)
    match_range = re.match(pattern_range, user_input, re.IGNORECASE)

    if match_all:
        return {
            "type": "all",
            "query": match_all.group(1)
        }

    if match_range:
        return {
            "type": "range",
            "start": match_range.group(1),
            "end": match_range.group(2),
            "query": match_range.group(3)
        }

    return None

def search_memory(query, messages, top_k = 10):
    results = []
    query_terms = query.lower().split()

    for msg in messages:
        content = msg.get("content", "").lower()

        # Match ALL words (better precision)
        if all(term in content for term in query_terms):
            score = sum(term in content for term in query_terms)
            results.append((score, msg))

    # Sort by score (higher first)
    results.sort(key=lambda x: x[0], reverse=True)

    return [m[1] for m in results[:top_k]]

def format_memory(memory):
    return "\n".join(m["content"] for m in memory)

# ------------------ MAIN ------------------
chat_file = create_chat_file()

conversation = []

print("Suhas's Coding Assistant (🤖): Hello! (type 'exit' to quit)")
print("Commands:")
print("/load all: <query>")
print("/load range from YYYY-MM-DD to YYYY-MM-DD: <query>")
print("=" * 50)

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("Suhas's Coding Assistant (🤖): Goodbye!")
        break

    # Detect memory command
    command = parse_command(user_input)

    # Default behavior (normal chat)
    original_query = user_input
    context_block = ""
    memory = None

    # Base messages always start with system prompt
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    # ------------------ MEMORY INJECTION ------------------
    if command:
        print("\n🔍 Searching past conversations...\n")

        if command["type"] == "all":
            all_chats = load_all_chats()
        else:
            all_chats = load_chats_by_date_range(
                command["start"],
                command["end"]
            )

        memory = search_memory(command["query"], all_chats)

        print("\nDEBUG MEMORY RESULTS:")
        for m in memory:
            print(m)
        print("\n")

        if memory:
            context_block = (
                "You are a retrieval system.\n"
                "Your job is to extract exact relevant information from the context.\n\n"

                "Rules:\n"
                "- Return exact matching lines from context\n"
                "- Do NOT summarize\n"
                "- Do NOT generate new content\n"
                "- If multiple matches exist, return them\n"
                "- If nothing matches, say: Not found in memory\n\n"

                "Context:\n"
                + format_memory(memory)
                + "\n\nQuery:\n"
            )
        else:
            print("⚠️ No relevant memory found.\n")

        # Override query only for memory case
        original_query = command["query"]

    # ------------------ CURRENT CHAT ------------------
    conversation.append({ "role": "user", "content": original_query })

    # Send enriched query to model
    model_query = context_block + original_query if memory else original_query

    # Trim history
    if len(conversation) > MAX_HISTORY:
        conversation = conversation[-MAX_HISTORY:]

    messages += conversation[:-1] + [{"role": "user", "content": model_query}]

    try:
        response = client.chat.completions.create(
            model = 'phi4-mini',
            messages = messages,
            stream = True  # Enable streaming responses
        )

        full_response = ""

        print("🤖: ", end = "", flush = True)

        # Stream output response
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            print(content, end = "", flush = True)  # Print the streamed content in real-time
            full_response += content

        print("\n")  # Print a newline after the full response is received

        conversation.append({ "role": "assistant", "content": full_response })  # Add the assistant's response to the conversation history
        
        save_conversation(chat_file, conversation)  # Save the conversation history to a file
    except Exception as e:
        print(f"Error: {e}")
        conversation.pop()  # Remove the last unanswered user message from history if there's an error