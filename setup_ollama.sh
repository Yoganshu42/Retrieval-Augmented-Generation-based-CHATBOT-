#!/bin/bash

# Setup script for RAG Chatbot with Ollama integration
# This script helps set up Ollama and download a recommended model

set -e

echo "🦙 RAG Chatbot - Ollama Setup Script"
echo "===================================="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "📦 Ollama not found. Installing Ollama..."
    
    # Detect OS and install Ollama
    if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
        curl -fsSL https://ollama.ai/install.sh | sh
    else
        echo "❌ Automatic installation not supported on this OS."
        echo "   Please download Ollama from: https://ollama.ai"
        exit 1
    fi
    
    echo "✅ Ollama installed successfully!"
else
    echo "✅ Ollama is already installed"
fi

# Start Ollama service (in background)
echo "🚀 Starting Ollama service..."
if ! pgrep -f "ollama serve" > /dev/null; then
    ollama serve &
    OLLAMA_PID=$!
    echo "✅ Ollama service started (PID: $OLLAMA_PID)"
    
    # Wait a moment for the service to start
    sleep 3
else
    echo "✅ Ollama service is already running"
fi

# Check if Ollama is responding
echo "🔍 Checking Ollama connection..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama is responding"
else
    echo "❌ Ollama is not responding. Please check the installation."
    exit 1
fi

# List current models
echo "📋 Checking available models..."
MODELS=$(ollama list | tail -n +2 | wc -l)
if [ "$MODELS" -gt 0 ]; then
    echo "✅ Found $MODELS model(s) already installed:"
    ollama list
else
    echo "⚠️  No models found"
fi

# Offer to download recommended model
echo ""
echo "🎯 Recommended Models for RAG Chatbot:"
echo "   1. llama3.2:3b (Default, ~2GB) - Balanced performance"
echo "   2. phi3:mini (~2GB) - Microsoft, efficient"  
echo "   3. gemma2:2b (~1.5GB) - Google, compact"
echo "   4. mistral:7b (~4GB) - High quality, larger"
echo "   5. Skip model download"
echo ""

read -p "Which model would you like to download? (1-5): " choice

case $choice in
    1)
        echo "📥 Downloading llama3.2:3b..."
        ollama pull llama3.2:3b
        ;;
    2)
        echo "📥 Downloading phi3:mini..."
        ollama pull phi3:mini
        ;;
    3)
        echo "📥 Downloading gemma2:2b..."
        ollama pull gemma2:2b
        ;;
    4)
        echo "📥 Downloading mistral:7b..."
        ollama pull mistral:7b
        ;;
    5)
        echo "⏭️  Skipping model download"
        ;;
    *)
        echo "❌ Invalid choice. Skipping model download."
        ;;
esac

# Test the setup
echo ""
echo "🧪 Testing setup..."
if python3 test_ollama_setup.py; then
    echo ""
    echo "🎉 Setup complete! Your RAG Chatbot is ready to use."
    echo ""
    echo "Next steps:"
    echo "1. Add your documents to the 'data/' directory"
    echo "2. Run the chatbot: streamlit run app.py"
    echo "3. Open http://localhost:8501 in your browser"
    echo ""
    echo "Useful Ollama commands:"
    echo "  ollama list          # List installed models"
    echo "  ollama pull <model>  # Download a new model"
    echo "  ollama rm <model>    # Remove a model"
    echo "  ollama ps            # Show running models"
else
    echo "❌ Setup test failed. Please check the error messages above."
    exit 1
fi