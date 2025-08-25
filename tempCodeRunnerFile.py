# for i, chunks in enumerate(sentences, 1):
#     print(f"{i}. Chunk: {chunks}\n")
from vector_db import search_faiss

def main():
    print("Hello from my-rag-project!")
    s = search_faiss()
    print(s)
