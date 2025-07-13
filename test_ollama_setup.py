#!/usr/bin/env python3
"""
Test script to verify Ollama integration is working correctly.
Run this script to check if your Ollama setup is configured properly.
"""

import sys
import os
sys.path.append('src')

from llm_handler import LLMHandler, SimpleLLM

def test_ollama_connection():
    """Test Ollama connection and model availability."""
    print("🦙 Testing Ollama Setup")
    print("=" * 50)
    
    try:
        # Initialize Ollama handler
        print("1. Initializing Ollama handler...")
        llm = LLMHandler()
        
        # Check health
        print("2. Checking Ollama server health...")
        if llm.check_health():
            print("   ✅ Ollama server is running")
        else:
            print("   ❌ Ollama server is not accessible")
            print("   💡 Make sure to run 'ollama serve' in another terminal")
            return False
        
        # List available models
        print("3. Checking available models...")
        models = llm.list_available_models()
        if models:
            print(f"   ✅ Found {len(models)} models:")
            for model in models[:5]:  # Show first 5 models
                print(f"      • {model}")
        else:
            print("   ⚠️  No models found")
            print("   💡 Try: ollama pull llama3.2:3b")
            return False
        
        # Test basic generation
        print("4. Testing response generation...")
        test_chunks = [
            {
                'content': 'The quick brown fox jumps over the lazy dog. This is a test document.',
                'filename': 'test.txt',
                'similarity_score': 0.95
            }
        ]
        
        response = llm.answer_query("What is in the test document?", test_chunks, stream=False)
        print(f"   ✅ Response generated: {response[:100]}...")
        
        # Test streaming
        print("5. Testing streaming response...")
        print("   🔄 Streaming response: ", end="", flush=True)
        for token in llm.answer_query("What is in the test document?", test_chunks, stream=True):
            print(token, end="", flush=True)
        print()
        
        print("\n🎉 All tests passed! Ollama integration is working correctly.")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_fallback_llm():
    """Test the fallback simple LLM."""
    print("\n🔧 Testing Fallback Simple LLM")
    print("=" * 50)
    
    try:
        simple_llm = SimpleLLM()
        
        test_chunks = [
            {
                'content': 'Our privacy policy protects user data and ensures confidentiality.',
                'filename': 'terms.txt',
                'similarity_score': 0.95
            }
        ]
        
        response = simple_llm.answer_query("What is your privacy policy?", test_chunks)
        print(f"✅ Simple LLM response: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with Simple LLM: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 RAG Chatbot - Ollama Integration Test")
    print("=" * 60)
    
    # Test Ollama
    ollama_works = test_ollama_connection()
    
    # Test fallback
    fallback_works = test_fallback_llm()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    print(f"Ollama Integration: {'✅ PASS' if ollama_works else '❌ FAIL'}")
    print(f"Fallback Simple LLM: {'✅ PASS' if fallback_works else '❌ FAIL'}")
    
    if ollama_works:
        print("\n🎯 Your setup is ready! You can now run:")
        print("   streamlit run app.py")
    elif fallback_works:
        print("\n⚠️  Ollama is not available, but fallback LLM works.")
        print("   The app will use simple rule-based responses.")
        print("   To enable Ollama:")
        print("     1. Install: curl -fsSL https://ollama.ai/install.sh | sh")
        print("     2. Start: ollama serve")
        print("     3. Pull model: ollama pull llama3.2:3b")
    else:
        print("\n❌ Setup issues detected. Please check your installation.")
    
    return ollama_works or fallback_works


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)