#!/usr/bin/env python3
"""
Verification script to ensure the migration from Transformers to Ollama is complete.
This script checks all components and configurations.
"""

import sys
import os
import json
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are available."""
    print("📦 Checking Dependencies...")
    
    required_packages = [
        'ollama', 'requests', 'streamlit', 'numpy', 
        'scikit-learn', 'sentence_transformers', 'faiss'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install -e .")
        return False
    
    return True


def check_project_structure():
    """Check if the project structure is correct."""
    print("\n📁 Checking Project Structure...")
    
    required_files = [
        'pyproject.toml',
        'requirements.txt', 
        'src/llm_handler.py',
        'src/rag_pipeline.py',
        'src/vector_store.py',
        'src/document_processor.py',
        'app.py',
        'test_ollama_setup.py',
        'setup_ollama.sh',
        'MIGRATION_GUIDE.md'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - Missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        return False
    
    return True


def check_configuration():
    """Check if configuration files are updated for Ollama."""
    print("\n⚙️  Checking Configuration...")
    
    # Check pyproject.toml
    try:
        with open('pyproject.toml', 'r') as f:
            content = f.read()
            
        if 'transformers' in content:
            print("   ⚠️  pyproject.toml still contains 'transformers' references")
            return False
        
        if 'ollama' in content:
            print("   ✅ pyproject.toml updated for Ollama")
        else:
            print("   ❌ pyproject.toml missing Ollama dependency")
            return False
            
    except Exception as e:
        print(f"   ❌ Error reading pyproject.toml: {e}")
        return False
    
    return True


def check_code_integration():
    """Check if the code has been properly updated."""
    print("\n🔍 Checking Code Integration...")
    
    # Check LLM handler
    try:
        sys.path.append('src')
        from llm_handler import LLMHandler, SimpleLLM
        
        # Try to create handlers (this will test imports)
        simple_llm = SimpleLLM()
        print("   ✅ SimpleLLM can be instantiated")
        
        # Test Ollama handler creation (may fail if Ollama not available)
        try:
            ollama_llm = LLMHandler()
            print("   ✅ Ollama LLMHandler can be instantiated")
        except Exception as e:
            print(f"   ⚠️  Ollama LLMHandler: {e}")
            print("      (This is expected if Ollama is not running)")
        
    except Exception as e:
        print(f"   ❌ Error importing handlers: {e}")
        return False
    
    # Check RAG pipeline
    try:
        from rag_pipeline import RAGPipeline
        print("   ✅ RAGPipeline can be imported")
        
        # Test pipeline creation with simple LLM
        pipeline = RAGPipeline(use_simple_llm=True)
        print("   ✅ RAGPipeline can be instantiated with Simple LLM")
        
    except Exception as e:
        print(f"   ❌ Error with RAGPipeline: {e}")
        return False
        
    return True


def main():
    """Main verification function."""
    print("🔍 RAG Chatbot - Migration Verification")
    print("=" * 50)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure), 
        ("Configuration", check_configuration),
        ("Code Integration", check_code_integration)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"   ❌ {check_name} check failed: {e}")
            results.append(False)
    
    # Summary
    print("\n📋 Verification Summary")
    print("=" * 30)
    
    all_passed = all(results)
    
    for i, (check_name, _) in enumerate(checks):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{check_name}: {status}")
    
    if all_passed:
        print("\n🎉 Migration Complete!")
        print("   Your RAG Chatbot has been successfully migrated to Ollama.")
        print("\n   Next steps:")
        print("   1. Run: ./setup_ollama.sh (to install Ollama and models)")
        print("   2. Run: python test_ollama_setup.py (to test)")
        print("   3. Run: streamlit run app.py (to start the chatbot)")
    else:
        print("\n⚠️  Migration Issues Detected")
        print("   Please review the failed checks above.")
        print("   Check MIGRATION_GUIDE.md for detailed instructions.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)