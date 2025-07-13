"""
Streamlit app for RAG-based chatbot.
Provides a user-friendly interface with real-time streaming responses.
"""

import streamlit as st
import time
from pathlib import Path
import sys

# Add src directory to path
sys.path.append('src')

from rag_pipeline import RAGPipeline
import os


# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .source-doc {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
        font-size: 0.8rem;
    }
    .stats-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_pipeline():
    """Initialize the RAG pipeline (cached for performance)."""
    try:
        # Try Ollama first, fallback to simple LLM if Ollama is not available
        pipeline = RAGPipeline(
            embedding_model="all-MiniLM-L6-v2",
            llm_model="llama3.2:3b",  # Default Ollama model
            vector_store_type="faiss",
            use_simple_llm=False,  # Try Ollama first
            ollama_base_url="http://localhost:11434"
        )
        
        # Check if Ollama is available
        if not pipeline.llm_handler.check_health():
            st.warning("Ollama server is not running. Falling back to simple LLM. To use Ollama, please start it with 'ollama serve'")
            # Reinitialize with simple LLM
            pipeline = RAGPipeline(
                embedding_model="all-MiniLM-L6-v2",
                vector_store_type="faiss",
                use_simple_llm=True
            )
        
        success = pipeline.initialize_from_documents(force_rebuild=True)
        
        if success:
            return pipeline
        else:
            st.error("Failed to initialize pipeline!")
            return None
            
    except Exception as e:
        st.error(f"Error initializing pipeline: {e}")
        # Try fallback to simple LLM
        try:
            st.info("Attempting to initialize with simple LLM as fallback...")
            pipeline = RAGPipeline(
                embedding_model="all-MiniLM-L6-v2",
                vector_store_type="faiss",
                use_simple_llm=True
            )
            success = pipeline.initialize_from_documents()
            if success:
                return pipeline
        except Exception as fallback_error:
            st.error(f"Fallback initialization also failed: {fallback_error}")
        
        return None


def display_message(message, is_user=True):
    """Display a chat message with styling."""
    css_class = "user-message" if is_user else "bot-message"
    icon = "👤" if is_user else "🤖"
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <strong>{icon} {'You' if is_user else 'Assistant'}:</strong>
        <div style="margin-top: 0.5rem;">{message}</div>
    </div>
    """, unsafe_allow_html=True)


def display_sources(sources):
    """Display source documents."""
    if sources:
        st.markdown("**📚 Sources:**")
        for i, source in enumerate(sources[:3]):  # Show top 3 sources
            with st.expander(f"Source {i+1}: {source['filename']} (Score: {source['similarity_score']:.3f})"):
                st.text(source['content'][:500] + "..." if len(source['content']) > 500 else source['content'])
                st.caption(f"Chunk: {source['chunk_id']} | Words: {source['word_count']}")


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 RAG-Powered Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize pipeline
    with st.spinner("Initializing RAG pipeline..."):
        pipeline = initialize_pipeline()
    
    if pipeline is None:
        st.error("Failed to initialize the chatbot. Please check your setup.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Pipeline stats
        stats = pipeline.get_stats()
        st.markdown('<div class="stats-container">', unsafe_allow_html=True)
        st.markdown("### 📊 Pipeline Statistics")
        st.write(f"**Model:** {stats['llm_model']}")
        st.write(f"**Embeddings:** {stats['embedding_model']}")
        st.write(f"**Vector Store:** {stats['vector_store_type'].upper()}")
        st.write(f"**Documents:** {stats['total_documents']}")
        st.write(f"**Chunks:** {stats['total_chunks']}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Settings
        st.header("🎛️ Settings")
        k_docs = st.slider("Documents to retrieve", 1, 10, 5)
        stream_response = st.checkbox("Stream responses", value=True)
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()
        
        # Reset pipeline button
        if st.button("🔄 Reset Pipeline", type="secondary"):
            st.cache_resource.clear()
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I'm your RAG-powered assistant. I can answer questions based on the documents in the knowledge base. What would you like to know?",
            "sources": []
        })
    
    # Display chat history
    for message in st.session_state.messages:
        display_message(message["content"], message["role"] == "user")
        if message["role"] == "assistant" and message.get("sources"):
            display_sources(message["sources"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about the documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
        display_message(prompt, True)
        
        # Generate response
        try:
            with st.spinner("Thinking..."):
                if stream_response:
                    # Streaming response
                    answer_stream, sources = pipeline.query(prompt, k=k_docs, stream=True)
                    
                    # Create placeholder for streaming
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    # Display bot icon
                    st.markdown("🤖 **Assistant:**")
                    
                    # Stream the response
                    for chunk in answer_stream:
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                        time.sleep(0.05)  # Small delay for visual effect
                    
                    # Final response without cursor
                    response_placeholder.markdown(full_response)
                    
                else:
                    # Non-streaming response
                    answer, sources = pipeline.query(prompt, k=k_docs, stream=False)
                    display_message(answer, False)
                    full_response = answer
                
                # Display sources
                display_sources(sources)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": sources
                })
        
        except Exception as e:
            st.error(f"Error generating response: {e}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I'm sorry, I encountered an error while processing your question. Please try again.",
                "sources": []
            })


if __name__ == "__main__":
    main()