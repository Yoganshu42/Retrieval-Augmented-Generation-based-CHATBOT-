# 🤖 RAG Chatbot (FAISS + Llama-3 + Streamlit)

A **Retrieval-Augmented Generation (RAG) chatbot** built with **FAISS**, **Ollama (Llama-3)**, and **Streamlit**.  
This chatbot can load documents, chunk them with overlap, embed them, store in FAISS, and answer questions grounded in the retrieved context.

---

## ✨ Features
- 📄 Document ingestion & cleaning (PDF, TXT, DOCX support)
- ✂️ Semantic chunking with overlap
- 🔎 Embedding with `sentence-transformers (all-MiniLM-L6-v2)`
- 📚 FAISS vector store (with cosine similarity)
- 🧠 Custom Ollama Llama-3 model with Modelfile (`rag_chatbot`)
- 💬 Streamlit UI with chat-like interface
  - **Enter** = Send  
  - **Shift+Enter** = New line
  - User message shown instantly, bot streams response with typing effect
- ⏱️ Response time tracking
- ⚡ Save/load FAISS index and chunks to disk for fast startup

---

## 📂 Project Structure

rag-chatbot/
│
├── app.py # Streamlit UI
├── rag_chatbot.Modelfile # Custom Ollama system prompt
├── rag_pipeline.py # RAG pipeline: retrieval → prompt → Ollama
├── vector_db.py # FAISS index build + search + persistence
├── data_chunking.py # Semantic chunking + embeddings
├── data_loading_and_cleaning.py# Document ingestion & cleaning
├── pyproject.toml # uv project dependencies
└── uv.lock # locked dependencies

yaml
Copy
Edit

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
2. Install dependencies with uv
bash
Copy
Edit
uv sync
3. Start Ollama
Make sure Ollama is running:

bash
Copy
Edit
ollama serve
4. Create custom model
Build your custom chatbot model with system prompt:

bash
Copy
Edit
ollama create rag_chatbot -f rag_chatbot.Modelfile
5. Run the chatbot
bash
Copy
Edit
uv run streamlit run app.py
Open your browser at http://localhost:8501.
```
📸 Screenshots
Chat Interface

User messages (blue box)

Bot messages (grey box with typing effect)

Sidebar with creator info & settings

🔧 Future Improvements

📚 Support for ChromaDB / pgvector

📝 Conversation memory

🎛️ Configurable chunk size & overlap from UI

📊 Evaluation dashboard

👨‍💻 Author

Yoganshu Sharma
🚀 RAG + Llama-3 + FAISS Enthusiast