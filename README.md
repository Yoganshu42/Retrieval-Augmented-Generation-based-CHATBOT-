# RAG-Based Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Python, featuring document processing, vector similarity search, and LLM-powered responses through a Streamlit interface.

## 🎯 Overview

This project implements a complete RAG pipeline that can:
- Process and chunk documents intelligently
- Create semantic embeddings using sentence transformers
- Store and search vectors using FAISS or ChromaDB
- Generate contextual responses using language models
- Provide real-time streaming responses via Streamlit

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Documents     │───▶│   Chunking &    │───▶│   Vector DB     │
│   (.txt files)  │    │   Embedding     │    │  (FAISS/Chroma) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐            │
│   Streamlit     │◀───│      LLM        │◀───────────┘
│   Interface     │    │   (Response     │    Query + Context
└─────────────────┘    │   Generation)   │
                       └─────────────────┘
```

## 📁 Project Structure

```
rag-based-chatbot/
├── data/                    # Input documents
│   └── AI Training Document.pdf
├── chunks/                  # Processed document chunks  
├── vectordb/               # Vector database files
├── notebooks/              # Jupyter notebooks (optional)
├── src/                    # Core modules
│   ├── document_processor.py  # Document loading & chunking
│   ├── vector_store.py        # Vector database operations
│   ├── llm_handler.py         # LLM integration
│   └── rag_pipeline.py        # Main RAG orchestration
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd rag-based-chatbot

# Create virtual environment (if not already done)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Your Documents

Place your documents (`.pdf` files) in the `data/` directory. The system will automatically process them.

### 3. Run the Application

```bash
streamlit run app.py
```

The application will:
1. Automatically process documents on first run
2. Create embeddings and build the vector database
3. Launch the chat interface at `http://localhost:8501`

## 🔧 Configuration Options

### Vector Store Types
- **FAISS**: Fast similarity search (default)

### LLM Options
- **Simple LLM**: Rule-based responses (default for demo)
- **Ollama Models**: mistral:7b, llama3.2:3b, phi3:mini, codellama:7b
- **Custom Models**: Configure in `src/llm_handler.py`

### Embedding Models
- `all-MiniLM-L6-v2` (default)
- `all-mpnet-base-v2`
- `bge-small-en`

## 📚 Core Components

### Document Processor (`src/document_processor.py`)
- Loads documents from the data directory
- Performs sentence-aware chunking (100-300 words)
- Adds overlapping context between chunks
- Handles text cleaning and normalization

### Vector Store (`src/vector_store.py`)
- Creates semantic embeddings using SentenceTransformers
- Supports both FAISS and ChromaDB backends
- Implements similarity search with scoring
- Handles persistence and loading

### LLM Handler (`src/llm_handler.py`)
- Integrates with Ollama
- Supports model quantization for GPU efficiency
- Implements streaming response generation
- Creates RAG-optimized prompts

### RAG Pipeline (`src/rag_pipeline.py`)
- Orchestrates the complete RAG workflow
- Handles initialization and configuration
- Provides unified query interface
- Manages pipeline statistics

## 🎨 Streamlit Interface Features

- **Real-time Chat**: Conversational interface with message history
- **Streaming Responses**: Token-by-token response generation
- **Source Citations**: Shows relevant document chunks used
- **Configuration Panel**: Adjust retrieval parameters
- **Pipeline Statistics**: Monitor system performance
- **Clear/Reset Functions**: Manage chat and pipeline state

## 📊 Usage Examples

### Basic Query
```python
from src.rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline(use_simple_llm=True)
pipeline.initialize_from_documents()

# Query
answer, sources = pipeline.query("What is your privacy policy?")
print(answer)
```

### Streaming Response
```python
# Streaming query
answer_stream, sources = pipeline.query(
    "Tell me about user responsibilities", 
    stream=True
)

for token in answer_stream:
    print(token, end="", flush=True)
```

## 🔍 Advanced Configuration

### Custom Embedding Model
```python
pipeline = RAGPipeline(
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    vector_store_type="chroma"
)
```


### Document Processing Parameters
```python
processor = DocumentProcessor(
    chunk_size=300,  # words per chunk
    overlap=75       # overlap between chunks
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Issues**
   - Use `use_simple_llm=True` for testing
   - Reduce `chunk_size` and `k` parameters
   - Enable model quantization

2. **Slow Performance**
   - Use FAISS instead of ChromaDB
   - Reduce number of retrieved documents
   - Use smaller embedding models

### Environment Variables
Create a `.env` file for configuration:
```env
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_STORE_TYPE=faiss
USE_SIMPLE_LLM=true
MAX_CHUNKS=1000
```

## 📝 Development

### Running Tests
```bash
# Test individual components
python src/document_processor.py
python src/vector_store.py
python src/rag_pipeline.py
```

### Adding New Documents
1. Place `.txt` files in the `data/` directory
2. Delete the `vectordb/` folder to force rebuild
3. Restart the application

### Customizing the LLM
Modify `src/llm_handler.py` to add new models or change prompt templates.

## 🎯 Assignment Completion Checklist

- ✅ Document preprocessing and chunking
- ✅ Semantic embedding generation
- ✅ Vector database (FAISS) integration
- ✅ LLM integration with prompt optimization
- ✅ Complete RAG pipeline implementation
- ✅ Streamlit interface with streaming
- ✅ Source document citations
- ✅ Real-time response generation
- ✅ Configuration management
- ✅ Comprehensive documentation

## 🚀 Demo

The application includes:
- Interactive chat interface
- Real-time streaming responses
- Source document highlighting
- Pipeline statistics dashboard
- Configurable retrieval parameters

### 📷 Screenshots and Demo Video

Please refer to the following for a demonstration of the chatbot:

- `screenshots/` directory: Includes images of the running application (e.g., input/output, streaming view).
- [Demo Video Link](https://your-demo-video-url.com) – A short video showing live interaction with the chatbot.

## 📄 License

This project is for educational purposes as part of an NLP/RAG assessment.

## 🤝 Contributing

This is an assessment project. For improvements or suggestions, please create an issue or submit a pull request.

---

**Note**: This implementation uses a simple rule-based LLM by default for demonstration purposes. For production use, configure it with proper transformer models and GPU acceleration.