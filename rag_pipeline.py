import ollama
from vector_db import search_faiss
from data_chunking import embed_sentences


def build_prompt(query, retrieved_chunks):
    context = "\n\n".join(
        [f"Source {i+1}: \n{chunk}" for i, (chunk,score) in enumerate(retrieved_chunks)]
    )
    return f"""
    Context:
    {context}

    Question:
    {query}

    Answer:
    """

def rag_answer(query, top_k = 5):
    # 1. Retrieve from FAISS
    retrieved = search_faiss(query, top_k = top_k)

    # 2. Build RAG Prompt
    prompt = build_prompt(query, retrieved)

    # 3. Response from Ollama Chat model
    response = ollama.chat(model = "rag_chatbot", messages = [
        {"role": "user", "content": prompt}
    ])

    return response["message"]["content"]



# # CLI Test

# if __name__ == "__main__":
#     query = input("Ask me something: ")
#     answer = rag_answer(query, top_k=5)
#     print("\n--- RAG Chatbot Answer ---\n") 
#     print(answer)