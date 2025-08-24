from data_loading_and_cleaning import cleaned_text
import numpy as np
import nltk
# Run this only once at first time only
# nltk.download('punkt')
# nltk.download('punkt_tab')
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def split_into_sentences(cleaned_text):
    # Splits data into sentence chunks
    return sent_tokenize(cleaned_text)
sentences = split_into_sentences(cleaned_text)

def semantic_chunking(sentences,chunk_size = 200, overlap = 50):
    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        words = sent.split()
        if current_len + len(words) <= chunk_size:
            current_chunk.append(sent)
            current_len += len(words)
        
        else:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

            overlap_words = chunk_text.split()[-overlap:]
            current_chunk = [" ".join(overlap_words),sent]
            current_len = len(overlap_words)+len(words)

    if current_chunk:
            chunks.append(" ".join(current_chunk))
        
    return chunks
    
chunks = semantic_chunking(sentences,chunk_size = 200, overlap = 50)
print(f"Generated: {len(chunks)} chunks.")

def embed_sentences(chunks):
     # Embedding the sentence chunks for converting to vector data
     model = SentenceTransformer('all-MiniLM-L6-v2')
     vecs = model.encode(chunks, normalize_embeddings = True) 
     return np.array(vecs) 
embeddings = embed_sentences(chunks)
print(f"Embeddings:{embeddings.shape}")

