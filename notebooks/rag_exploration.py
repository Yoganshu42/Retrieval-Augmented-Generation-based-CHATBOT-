"""
RAG Pipeline Exploration and Testing Notebook
This file can be run as a script or converted to Jupyter notebook format.
"""

import sys
sys.path.append('../src')

from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
import json


def test_document_processing():
    """Test document processing functionality."""
    print("="*60)
    print("TESTING DOCUMENT PROCESSING")
    print("="*60)
    
    processor = DocumentProcessor(chunk_size=150, overlap=30)
    
    # Load documents
    documents = processor.load_documents("../data")
    print(f"Loaded {len(documents)} documents")
    
    for doc in documents:
        print(f"- {doc['filename']}: {len(doc['content'].split())} words")
    
    # Chunk documents
    chunks = processor.chunk_documents(documents)
    print(f"\nCreated {len(chunks)} chunks")
    
    # Display sample chunks
    print("\nSample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1} ({chunk['chunk_id']}):")
        print(f"Words: {chunk['word_count']}")
        print(f"Content: {chunk['content'][:200]}...")
    
    return chunks


def test_vector_store(chunks):
    """Test vector store functionality."""
    print("\n" + "="*60)
    print("TESTING VECTOR STORE")
    print("="*60)
    
    # Test FAISS
    print("Testing FAISS Vector Store...")
    faiss_store = VectorStore(store_type="faiss")
    faiss_store.build_faiss_index(chunks, "../vectordb")
    
    # Test search
    test_queries = [
        "privacy policy",
        "user responsibilities", 
        "termination of account",
        "intellectual property"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = faiss_store.search(query, k=3)
        
        for i, result in enumerate(results):
            print(f"  {i+1}. Score: {result['similarity_score']:.3f}")
            print(f"     Source: {result['filename']}")
            print(f"     Content: {result['content'][:100]}...")
    
    return faiss_store


def test_rag_pipeline():
    """Test complete RAG pipeline."""
    print("\n" + "="*60)
    print("TESTING RAG PIPELINE")
    print("="*60)
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        embedding_model="all-MiniLM-L6-v2",
        vector_store_type="faiss",
        use_simple_llm=True
    )
    
    # Initialize from documents
    success = pipeline.initialize_from_documents("../data", force_rebuild=False)
    print(f"Pipeline initialization: {'Success' if success else 'Failed'}")
    
    if not success:
        return
    
    # Show stats
    stats = pipeline.get_stats()
    print("\nPipeline Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test queries
    test_queries = [
        "What is covered in your privacy policy?",
        "How can I terminate my account?",
        "What are my responsibilities as a user?",
        "Tell me about intellectual property rights",
        "What happens if I violate the terms?"
    ]
    
    print("\n" + "-"*40)
    print("TESTING QUERIES")
    print("-"*40)
    
    for query in test_queries:
        print(f"\nQ: {query}")
        
        # Get answer and sources
        answer, sources = pipeline.query(query, k=3)
        
        print(f"A: {answer}")
        print(f"Sources: {len(sources)} documents")
        
        # Show top source
        if sources:
            top_source = sources[0]
            print(f"Top source: {top_source['filename']} (score: {top_source['similarity_score']:.3f})")
    
    # Test streaming
    print("\n" + "-"*40)
    print("TESTING STREAMING RESPONSE")
    print("-"*40)
    
    query = "What are the main terms and conditions?"
    print(f"Query: {query}")
    print("Streaming answer: ", end="", flush=True)
    
    answer_stream, sources = pipeline.query(query, stream=True)
    for token in answer_stream:
        print(token, end="", flush=True)
    print()


def analyze_document_coverage():
    """Analyze how well the chunking covers the original documents."""
    print("\n" + "="*60)
    print("DOCUMENT COVERAGE ANALYSIS")
    print("="*60)
    
    processor = DocumentProcessor()
    documents = processor.load_documents("../data")
    chunks = processor.chunk_documents(documents)
    
    # Analyze coverage
    total_words = sum(len(doc['content'].split()) for doc in documents)
    chunk_words = sum(chunk['word_count'] for chunk in chunks)
    
    print(f"Original documents: {total_words} words")
    print(f"Total in chunks: {chunk_words} words")
    print(f"Coverage ratio: {chunk_words/total_words:.2f}")
    
    # Analyze chunk distribution
    chunk_sizes = [chunk['word_count'] for chunk in chunks]
    print(f"\nChunk size statistics:")
    print(f"  Min: {min(chunk_sizes)} words")
    print(f"  Max: {max(chunk_sizes)} words")
    print(f"  Average: {sum(chunk_sizes)/len(chunk_sizes):.1f} words")
    
    # Analyze by document
    for doc in documents:
        doc_chunks = [c for c in chunks if c['filename'] == doc['filename']]
        print(f"\n{doc['filename']}:")
        print(f"  Original: {len(doc['content'].split())} words")
        print(f"  Chunks: {len(doc_chunks)}")
        print(f"  Chunk words: {sum(c['word_count'] for c in doc_chunks)}")


def main():
    """Run all tests."""
    print("RAG PIPELINE EXPLORATION AND TESTING")
    print("="*60)
    
    # Test components
    chunks = test_document_processing()
    vector_store = test_vector_store(chunks)
    test_rag_pipeline()
    analyze_document_coverage()
    
    print("\n" + "="*60)
    print("TESTING COMPLETE!")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Run 'streamlit run ../app.py' to launch the web interface")
    print("2. Try different queries in the chat interface")
    print("3. Experiment with different models and parameters")
    print("4. Add more documents to the data/ folder")


if __name__ == "__main__":
    main()