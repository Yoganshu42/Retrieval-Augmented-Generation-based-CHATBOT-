#!/usr/bin/env python3
"""
Setup test script for RAG-based chatbot.
Run this to verify everything is working correctly.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    try:
        import tiktoken
        print("✅ Tiktoken imported successfully")
    except ImportError as e:
        print(f"❌ Tiktoken import failed: {e}")
        return False
    
    # Test optional Streamlit import
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"⚠️ Streamlit import failed (optional): {e}")
        print("   You can still run the simple pipeline without Streamlit")
    
    return True


def test_directory_structure():
    """Test if directory structure is correct."""
    print("\nTesting directory structure...")
    
    required_dirs = ['data', 'src', 'chunks', 'vectordb', 'notebooks']
    required_files = [
        'app.py',
        'requirements.txt', 
        'README.md',
        'src/document_processor.py',
        'src/vector_store.py',
        'src/llm_handler.py',
        'src/rag_pipeline.py',
        'data/sample_document.txt'
    ]
    
    # Check directories
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ Directory '{dir_name}' exists")
        else:
            print(f"❌ Directory '{dir_name}' missing")
            return False
    
    # Check files
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"✅ File '{file_name}' exists")
        else:
            print(f"❌ File '{file_name}' missing")
            return False
    
    return True


def test_src_imports():
    """Test if src modules can be imported."""
    print("\nTesting src module imports...")
    
    # Add src to path
    sys.path.append('src')
    
    try:
        from document_processor import DocumentProcessor
        print("✅ DocumentProcessor imported successfully")
    except ImportError as e:
        print(f"❌ DocumentProcessor import failed: {e}")
        return False
    
    try:
        from simple_vector_store import SimpleVectorStore
        print("✅ SimpleVectorStore imported successfully")
    except ImportError as e:
        print(f"❌ SimpleVectorStore import failed: {e}")
        return False
    
    try:
        from llm_handler import SimpleLLM
        print("✅ SimpleLLM imported successfully")
    except ImportError as e:
        print(f"❌ SimpleLLM import failed: {e}")
        return False
    
    try:
        from simple_rag_pipeline import SimpleRAGPipeline
        print("✅ SimpleRAGPipeline imported successfully")
    except ImportError as e:
        print(f"❌ SimpleRAGPipeline import failed: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality of the pipeline."""
    print("\nTesting basic functionality...")
    
    try:
        sys.path.append('src')
        from document_processor import DocumentProcessor
        
        # Test document processing
        processor = DocumentProcessor()
        documents = processor.load_documents("data")
        
        if not documents:
            print("❌ No documents found in data/ directory")
            return False
        
        print(f"✅ Found {len(documents)} documents")
        
        # Test chunking
        chunks = processor.chunk_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_simple_pipeline():
    """Test the pipeline with simple LLM."""
    print("\nTesting simple RAG pipeline...")
    
    try:
        sys.path.append('src')
        from simple_rag_pipeline import SimpleRAGPipeline
        
        # Initialize simple pipeline
        pipeline = SimpleRAGPipeline()
        
        # Quick initialization test
        success = pipeline.initialize_from_documents()
        
        if success:
            print("✅ Simple RAG pipeline initialized successfully")
            
            # Quick query test
            answer, sources = pipeline.query("What is the privacy policy?", k=2)
            print(f"✅ Query test successful - found {len(sources)} sources")
            print(f"   Sample answer: {answer[:100]}...")
            
            return True
        else:
            print("❌ Simple RAG pipeline initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("RAG-BASED CHATBOT SETUP TEST")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Directory Structure", test_directory_structure), 
        ("Src Module Imports", test_src_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Simple Pipeline", test_simple_pipeline)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "="*50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Run: streamlit run app.py")
        print("2. Open http://localhost:8501 in your browser")
        print("3. Start chatting with your RAG bot!")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Check file permissions")
        print("- Verify Python version (3.8+)")
    
    print("="*50)


if __name__ == "__main__":
    main()