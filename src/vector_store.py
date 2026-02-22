"""
Vector store module for RAG chatbot.
Handles embedding generation and vector database operations.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", store_type: str = "faiss"):
        
        # Initialize vector store.
        self.embedding_model_name = embedding_model
        self.store_type = store_type
        self.model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize vector store
        if store_type == "faiss":
            self.index = None
            self.chunks_metadata = []
        
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        
        # Create embeddings for a list of texts.
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings
    
    def build_faiss_index(self, chunks: List[Dict[str, str]], save_path: str = "vectordb"):
        
        # Build FAISS index from document chunks.
        print("Creating embeddings...")
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.create_embeddings(texts)
        
        print("Building FAISS index...")
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.chunks_metadata = chunks
        
        # Save index and metadata
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        faiss.write_index(self.index, str(save_path / "faiss_index.index"))
        
        with open(save_path / "chunks_metadata.pkl", "wb") as f:
            pickle.dump(self.chunks_metadata, f)
        
        print(f"FAISS index saved with {self.index.ntotal} vectors")
    
    def load_faiss_index(self, load_path: str = "vectordb"):
        # Load FAISS index and metadata from disk.
        load_path = Path(load_path)
        
        # Load index
        self.index = faiss.read_index(str(load_path / "faiss_index.index"))
        
        # Load metadata
        with open(load_path / "chunks_metadata.pkl", "rb") as f:
            self.chunks_metadata = pickle.load(f)
        
        print(f"FAISS index loaded with {self.index.ntotal} vectors")
    
    def search_faiss(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        
        # Search FAISS index for similar documents.
    
        if self.index is None:
            raise ValueError("FAISS index not loaded. Call load_faiss_index() first.")
        
        # Create query embedding
        query_embedding = self.create_embeddings([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks_metadata):
                result = self.chunks_metadata[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        
        # Search for similar documents using the configured store type.
        if self.store_type == "faiss":
            return self.search_faiss(query, k)
        else:
            raise ValueError(f"Unsupported store type: {self.store_type}")


def main():
    # Test the vector store
    from document_processor import DocumentProcessor
    
    # Process documents first
    processor = DocumentProcessor()
    documents = processor.load_documents("data")
    chunks = processor.chunk_documents(documents)
    
    # Test FAISS
    print("Testing FAISS vector store...")
    faiss_store = VectorStore(store_type="faiss")
    faiss_store.build_faiss_index(chunks)
    
    # Test search
    results = faiss_store.search("privacy policy", k=3)
    print("\nFAISS Search Results:")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result['similarity_score']:.3f}")
        print(f"   Content: {result['content'][:100]}...")
        print()


if __name__ == "__main__":
    main()
