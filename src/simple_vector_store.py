"""
Simple vector store using sklearn for similarity search.
This is a simplified version that doesn't require FAISS.
"""

import pickle
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimpleVectorStore:
    def __init__(self):
        """Initialize simple vector store using TF-IDF."""
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.vectors = None
        self.chunks_metadata = []
        self.is_built = False
    
    def build_index(self, chunks: List[Dict[str, str]], save_path: str = "vectordb"):
        """
        Build vector index from document chunks.
        
        Args:
            chunks: List of document chunks
            save_path: Path to save the index
        """
        print("Creating TF-IDF vectors...")
        texts = [chunk['content'] for chunk in chunks]
        
        # Create TF-IDF vectors
        self.vectors = self.vectorizer.fit_transform(texts)
        self.chunks_metadata = chunks
        self.is_built = True
        
        # Save index and metadata
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        # Save vectorizer and vectors
        with open(save_path / "vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        
        with open(save_path / "vectors.pkl", "wb") as f:
            pickle.dump(self.vectors, f)
        
        with open(save_path / "chunks_metadata.pkl", "wb") as f:
            pickle.dump(self.chunks_metadata, f)
        
        print(f"Vector index saved with {len(chunks)} documents")
    
    def load_index(self, load_path: str = "vectordb"):
        """
        Load vector index from disk.
        
        Args:
            load_path: Path to load the index from
        """
        load_path = Path(load_path)
        
        # Load vectorizer
        with open(load_path / "vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        
        # Load vectors
        with open(load_path / "vectors.pkl", "rb") as f:
            self.vectors = pickle.load(f)
        
        # Load metadata
        with open(load_path / "chunks_metadata.pkl", "rb") as f:
            self.chunks_metadata = pickle.load(f)
        
        self.is_built = True
        print(f"Vector index loaded with {len(self.chunks_metadata)} documents")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        """
        Search for similar documents.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of similar documents with scores
        """
        if not self.is_built:
            raise ValueError("Index not built. Call build_index() or load_index() first.")
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        # Get top k results
        top_indices = similarities.argsort()[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if idx < len(self.chunks_metadata):
                result = self.chunks_metadata[idx].copy()
                result['similarity_score'] = float(similarities[idx])
                results.append(result)
        
        return results


def main():
    """Test the simple vector store."""
    from document_processor import DocumentProcessor
    
    # Process documents first
    processor = DocumentProcessor()
    documents = processor.load_documents("../data")
    chunks = processor.chunk_documents(documents)
    
    # Test simple vector store
    print("Testing Simple Vector Store...")
    store = SimpleVectorStore()
    store.build_index(chunks)
    
    # Test search
    results = store.search("privacy policy", k=3)
    print("\nSearch Results:")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result['similarity_score']:.3f}")
        print(f"   Content: {result['content'][:100]}...")
        print()


if __name__ == "__main__":
    main()