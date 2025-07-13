"""
RAG Pipeline module for the chatbot.
Orchestrates document retrieval and response generation.
"""

import os
from typing import List, Dict, Iterator, Optional, Tuple
from pathlib import Path
import json
from document_processor import DocumentProcessor
from vector_store import VectorStore
from llm_handler import LLMHandler, SimpleLLM


class RAGPipeline:
    def __init__(
        self, 
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama3.2:3b",
        vector_store_type: str = "faiss",
        use_simple_llm: bool = False,
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            embedding_model: Name of the embedding model
            llm_model: Name of the Ollama LLM model
            vector_store_type: Type of vector store ("faiss" or "chroma")
            use_simple_llm: Whether to use simple rule-based LLM
            ollama_base_url: Base URL for Ollama API
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.vector_store_type = vector_store_type
        self.use_simple_llm = use_simple_llm
        self.ollama_base_url = ollama_base_url
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore(embedding_model, vector_store_type)
        
        if use_simple_llm:
            self.llm_handler = SimpleLLM()
        else:
            self.llm_handler = LLMHandler(llm_model, ollama_base_url)
        
        self.is_initialized = False
        self.stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'vector_store_type': vector_store_type,
            'embedding_model': embedding_model,
            'llm_model': llm_model if not use_simple_llm else "Simple Rule-based Model"
        }
    
    def initialize_from_documents(
        self, 
        data_dir: str = "data", 
        force_rebuild: bool = False
    ) -> bool:
        """
        Initialize the pipeline by processing documents and building vector store.
        
        Args:
            data_dir: Directory containing documents
            force_rebuild: Whether to force rebuilding even if indexes exist
            
        Returns:
            Success status
        """
        try:
            print("Initializing RAG Pipeline...")
            
            # Check if we need to rebuild
            vector_db_path = Path("vectordb")
            if not force_rebuild and self._check_existing_indexes():
                print("Loading existing vector store...")
                return self._load_existing_pipeline()
            
            # Step 1: Load and process documents
            print("Step 1: Loading documents...")
            documents = self.document_processor.load_documents(data_dir)
            if not documents:
                print("No documents found!")
                return False
            
            self.stats['total_documents'] = len(documents)
            print(f"Loaded {len(documents)} documents")
            
            # Step 2: Chunk documents
            print("Step 2: Chunking documents...")
            chunks = self.document_processor.chunk_documents(documents)
            self.stats['total_chunks'] = len(chunks)
            print(f"Created {len(chunks)} chunks")
            
            # Save chunks
            self.document_processor.save_chunks(chunks, "chunks")
            
            # Step 3: Build vector store
            print("Step 3: Building vector store...")
            if self.vector_store_type == "faiss":
                self.vector_store.build_faiss_index(chunks)
            elif self.vector_store_type == "chroma":
                self.vector_store.build_chroma_index(chunks)
            
            # Step 4: Save pipeline configuration
            self._save_pipeline_config()
            
            self.is_initialized = True
            print("RAG Pipeline initialized successfully!")
            return True
            
        except Exception as e:
            print(f"Error initializing pipeline: {e}")
            return False
    
    def _check_existing_indexes(self) -> bool:
        """Check if vector indexes already exist."""
        vector_db_path = Path("vectordb")
        
        if self.vector_store_type == "faiss":
            return (vector_db_path / "faiss_index.index").exists()
        elif self.vector_store_type == "chroma":
            return (vector_db_path / "chroma").exists()
        
        return False
    
    def _load_existing_pipeline(self) -> bool:
        """Load existing pipeline from saved indexes."""
        try:
            # Load vector store
            if self.vector_store_type == "faiss":
                self.vector_store.load_faiss_index()
            elif self.vector_store_type == "chroma":
                # ChromaDB loads automatically
                self.vector_store.collection = self.vector_store.chroma_client.get_collection("rag_collection")
            
            # Load configuration
            config_path = Path("vectordb") / "pipeline_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.stats.update(json.load(f))
            
            self.is_initialized = True
            print("Existing pipeline loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading existing pipeline: {e}")
            return False
    
    def _save_pipeline_config(self):
        """Save pipeline configuration."""
        config_path = Path("vectordb") / "pipeline_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def retrieve_documents(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Call initialize_from_documents() first.")
        
        return self.vector_store.search(query, k)
    
    def generate_answer(
        self, 
        query: str, 
        retrieved_docs: List[Dict[str, str]], 
        stream: bool = False
    ) -> str | Iterator[str]:
        """
        Generate answer using LLM.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            stream: Whether to stream response
            
        Returns:
            Generated answer
        """
        return self.llm_handler.answer_query(query, retrieved_docs, stream)
    
    def query(
        self, 
        user_query: str, 
        k: int = 5, 
        stream: bool = False
    ) -> Tuple[str | Iterator[str], List[Dict[str, str]]]:
        """
        Complete RAG query pipeline.
        
        Args:
            user_query: User's question
            k: Number of documents to retrieve
            stream: Whether to stream response
            
        Returns:
            Tuple of (answer, source_documents)
        """
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Call initialize_from_documents() first.")
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(user_query, k)
        
        # Step 2: Generate answer
        answer = self.generate_answer(user_query, retrieved_docs, stream)
        
        return answer, retrieved_docs
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return self.stats.copy()
    
    def reset_pipeline(self):
        """Reset the pipeline (useful for testing)."""
        self.is_initialized = False
        
        # Clean up vector store
        vector_db_path = Path("vectordb")
        if vector_db_path.exists():
            import shutil
            shutil.rmtree(vector_db_path)
        
        # Clean up chunks
        chunks_path = Path("chunks")
        if chunks_path.exists():
            import shutil
            shutil.rmtree(chunks_path)


def main():
    """Test the RAG pipeline."""
    print("Testing RAG Pipeline...")
    
    # Initialize pipeline with simple LLM for testing
    pipeline = RAGPipeline(use_simple_llm=True)
    
    # Initialize from documents
    success = pipeline.initialize_from_documents(force_rebuild=True)
    
    if success:
        print("\nTesting queries...")
        
        # Test queries
        test_queries = [
            "What is your privacy policy?",
            "How can I terminate my account?",
            "What are the user responsibilities?",
            "Tell me about intellectual property rights."
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            answer, sources = pipeline.query(query, k=3)
            print(f"Answer: {answer}")
            print(f"Sources: {len(sources)} documents retrieved")
            
            # Show top source
            if sources:
                print(f"Top source: {sources[0]['filename']} (score: {sources[0]['similarity_score']:.3f})")
        
        # Test streaming
        print("\n" + "="*50)
        print("Testing streaming response...")
        query = "What happens if I violate the terms?"
        answer_stream, sources = pipeline.query(query, stream=True)
        
        print(f"Query: {query}")
        print("Streaming answer: ", end="", flush=True)
        for token in answer_stream:
            print(token, end="", flush=True)
        print()
        
        # Show stats
        print("\n" + "="*50)
        print("Pipeline Statistics:")
        stats = pipeline.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    else:
        print("Failed to initialize pipeline!")


if __name__ == "__main__":
    main()