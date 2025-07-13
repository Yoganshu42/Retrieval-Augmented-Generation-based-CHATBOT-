# Migration Guide: From Transformers to Ollama

This guide explains the changes made to migrate from Hugging Face Transformers to Ollama for LLM integration.

## 🔄 What Changed

### Dependencies
**Before (Transformers):**
```toml
"transformers>=4.30.0"
"torch>=2.0.0"
"bitsandbytes>=0.41.0"
"accelerate>=0.20.0"
```

**After (Ollama):**
```toml
"ollama>=0.1.7"
"requests>=2.31.0"
```

### LLM Handler
**Before:** Complex setup with GPU management, quantization, and model loading
**After:** Simple API calls to local Ollama server

### Model Configuration
**Before:**
```python
llm_handler = LLMHandler("microsoft/DialoGPT-medium", use_quantization=True)
```

**After:**
```python
llm_handler = LLMHandler("llama3.2:3b", "http://localhost:11434")
```

## 🚀 Benefits of Ollama

1. **Simplified Setup**: No need to manage GPU memory, quantization, or model loading
2. **Better Performance**: Optimized inference engine
3. **Model Management**: Easy model pulling and switching with `ollama pull/list/rm`
4. **Lower Resource Usage**: More efficient memory management
5. **Wider Model Support**: Access to Llama, Mistral, CodeLlama, Phi, Gemma, and more

## 🛠️ How to Use

### 1. Install Ollama
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai
```

### 2. Start Ollama Server
```bash
ollama serve
```

### 3. Pull a Model
```bash
# Lightweight (recommended for most users)
ollama pull llama3.2:3b      # ~2GB
ollama pull phi3:mini        # ~2GB

# Better quality (more resources needed)  
ollama pull llama3.1:8b      # ~4.7GB
ollama pull mistral:7b       # ~4.1GB
```

### 4. Run the Application
```bash
streamlit run app.py
```

The app will automatically:
- Detect if Ollama is running
- Use Ollama if available
- Fall back to simple LLM if Ollama is unavailable

## 🔧 Configuration Options

### Available Models
- `llama3.2:3b` - Default, balanced performance/size
- `llama3.2:1b` - Very lightweight
- `phi3:mini` - Microsoft Phi-3, efficient
- `gemma2:2b` - Google Gemma, compact
- `mistral:7b` - High quality, larger size
- `codellama:7b` - Optimized for code

### Custom Configuration
```python
pipeline = RAGPipeline(
    llm_model="mistral:7b",                    # Different model
    ollama_base_url="http://localhost:11434",  # Custom Ollama URL
    use_simple_llm=False                       # Force Ollama usage
)
```

## 🐛 Troubleshooting

### Ollama Not Found
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve
```

### No Models Available
```bash
# List models
ollama list

# Pull a model
ollama pull llama3.2:3b
```

### Memory Issues
- Use smaller models: `phi3:mini`, `gemma2:2b`
- Stop unused models: `ollama rm model-name`
- Monitor usage: `ollama ps`

## 🔄 Fallback Behavior

The application includes intelligent fallback:

1. **First**: Try to connect to Ollama
2. **If Ollama unavailable**: Use simple rule-based LLM
3. **User notification**: Clear indication of which mode is active

This ensures the application always works, even without Ollama setup.

## 📈 Performance Comparison

| Aspect | Transformers | Ollama |
|--------|-------------|---------|
| Setup Complexity | High | Low |
| Memory Usage | Higher | Lower |
| GPU Management | Manual | Automatic |
| Model Switching | Restart required | Instant |
| Performance | Good | Better |
| Resource Efficiency | Moderate | High |

## 🎯 Next Steps

1. **Test Setup**: Run `python test_ollama_setup.py`
2. **Choose Model**: Start with `llama3.2:3b` for balanced performance
3. **Monitor Resources**: Use `ollama ps` to check running models
4. **Experiment**: Try different models for your specific use case

For more information, visit [Ollama Documentation](https://ollama.ai/docs).