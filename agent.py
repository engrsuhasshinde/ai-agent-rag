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
            current_word_count + word_count > max_words and current_chunk  # Avoid creating empty chunks
        ):
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# =============================================
# EMBEDDINGS
# =============================================

def get_embedding(text):
    response = client.embeddings.create(
        model = EMBEDDING_MODEL,
        input = text
    )

    return response.data[0].embedding

# =============================================
# STORE MESSAGE IN CHROMADB
# =============================================

def save_message(message_id, role, text):
    chunks = split_into_chunks(text)

    for index, chunk in enumerate(chunks):
        chunk_id = f"{message_id}_chunk_{index}"

        collection.upsert(
            ids = [chunk_id],  # unique, stable ID for chunk
            embeddings = [get_embedding(chunk)],  # embedding vector for chunk
            documents = [chunk],  # the actual text content of the chunk
            metadatas = [{
                "role": role  # metadata to indicate if it's from user or assistant
            }]
        )

# =============================================
# SEMANTIC SEARCH
# =============================================

def find_relevant_context(query, top_k = TOP_K):
    if collection.count() == 0:
        return ""  # No data in collection, return empty context

    results = collection.query(
        query_embeddings = [get_embedding(query)],
        n_results = min(top_k, collection.count()),
        include = ["documents", "distances"]
    )

    relevant_chunks = []
    documents = results["documents"][0]  # List of retrieved document chunks
    distances = results["distances"][0]  # Corresponding similarity scores

    for document, distance in zip(documents, distances):
        similarity = 1 - distance  # Convert distance to similarity score

        if similarity >= SIMILARITY_THRESHOLD:  # Ignore weak/irrelevant matches
            relevant_chunks.append(document)

    return "\n".join(relevant_chunks)

# =============================================
# BUILD PROMPT WITH CONTEXT
# =============================================
# Inject relevant context into the system prompt so LLM can use it to generate informed responses

def build_system_prompt(context):
    if not context:
        return SYSTEM_PROMPT  # No context, return base system prompt

    return f"""{SYSTEM_PROMPT}

Relevant context from past conversation:

{context}

Use the context above only if it helps answer the question.
"""

# =============================================
# MAIN CONVERSATION LOOP
# =============================================

conversation = []
message_counter = 0

print("Suhas's Coding Assistant (🤖): Hello! (type 'exit' to quit)")
print("=" * 50)

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue  # Skip empty inputs

    if user_input.lower() == "exit":
        print("Suhas's Coding Assistant (🤖): Goodbye!")
        break

    # Retrieve semantic context from ChromaDB based on user input
    context = find_relevant_context(user_input)

    # Build final system prompt with relevant context
    system_prompt = build_system_prompt(context)

    # Add user message to conversation and call LLM
    conversation.append({
        "role": "user",
        "content": user_input
    })

    try:
        response = client.chat.completions.create(
            model = LLM_MODEL,
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
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

        # Save assistant response in conversation
        conversation.append({
            "role": "assistant",
            "content": full_response
        })

        # Store embeddings for both user message and assistant response in vector database
        save_message(
            f"message_{message_counter}",
            "user",
            user_input
        )

        save_message(
            f"message_{message_counter}",
            "assistant",
            full_response
        )

        message_counter += 1
    except Exception as e:
        print(f"Error: {e}")
        conversation.pop()  # Remove the last unanswered user message from history if there's an error