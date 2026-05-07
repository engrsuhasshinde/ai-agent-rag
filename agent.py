from openai import OpenAI
import chromadb
import re

# =============================================
# CONFIG / CONSTANTS
# =============================================

BASE_URL = "http://localhost:11434/v1"
API_KEY = "ollama"

LLM_MODEL = "phi4-mini"
EMBEDDING_MODEL = "nomic-embed-text"

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "chat_memory"  # Name of ChromaDB collection

CHUNK_SIZE = 60  # Number of words per chunk when splitting messages for embedding
TOP_K = 3  # Number of top similar chunks to retrieve per query
SIMILARITY_THRESHOLD = 0.30  # Minimum similarity score to consider a chunk relevant

SYSTEM_PROMPT = (
    "You are an expert coding assistant."
)

# =============================================
# OPENAI CLIENT
# =============================================

client = OpenAI(
    base_url = BASE_URL,
    api_key = API_KEY
)

# =============================================
# CHROMADB SETUP
# =============================================

db = chromadb.PersistentClient(path = CHROMA_DB_PATH)
collection = db.get_or_create_collection(
    name = COLLECTION_NAME,
    metadata = {
        "hnsw:space": "cosine"  # Use cosine similarity for embeddings
    }
)

# =============================================
# TEXT CHUNKING
# =============================================

def split_into_chunks(text, max_words = CHUNK_SIZE):
    sentences = re.split(r'(?<=[.!?]) +', text.strip())  # Split text into sentences
    sentences = [s for s in sentences if s.strip()]  # Remove empty sentences

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        if (
            current_word_count + word_count > max_words and current_chunk
        ):
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

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
            model = LLM_MODEL,
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }
            ] + conversation,
            stream = True  # Enables streaming responses
        )

        print("\n🤖: ", end = "", flush = True)

        # Stream output response
        full_response = ""
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            print(content, end = "", flush = True)
            full_response += content

        print("\n")  # Print a newline after the full response is received

        conversation.append({ "role": "assistant", "content": full_response })  # Add the assistant's response to the conversation history
    except Exception as e:
        print(f"Error: {e}")
        conversation.pop()  # Remove the last unanswered user message from history if there's an error