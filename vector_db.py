import faiss
import numpy as np
from data_chunking import embed_sentences, embeddings, chunks
from sklearn.metrics.pairwise import cosine_similarity 

d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
emb_norms = embeddings/np.linalg.norm(embeddings, axis= 1, keepdims = True) # Normalize
index.add(emb_norms.astype("float32"))

def search_faiss(query, top_k = 5):
    # Embed the query 
    q_emb = embed_sentences([query])
    q_emb = q_emb/np.linalg.norm(q_emb, axis = 1, keepdims = True) # Normalize

    sims, idx = index.search(q_emb.astype("float32"), top_k)
    results = [(chunks[i], float(sims[0][j])) for j,i in enumerate(idx[0])]
    return results



## For Testing the working of FAISS Index

# print(f"fais index contains: {index.ntotal} words")

# query = "What does the document say about regulations?"
# q_emb = embed_sentences([query])
# print("Query embedding shape:", q_emb.shape)

# results = search_faiss(query, top_k=5)
# print("Results found:", len(results))

# for text, score in results:
#     print(f"[Score: {score:.4f}] {text[:200]}...\n")
