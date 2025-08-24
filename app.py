import streamlit as st
import time
from rag_pipeline import rag_answer   # your RAG pipeline function

# Page config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")

# Custom CSS
st.markdown("""
<style>
    .creator-box {
        border-left: 4px solid #4CAF50;
        padding-left: 10px;
        margin-top: 10px;
        font-size: 0.9rem;
        color: #333;
    }
    .user-msg {
        background-color: #e3f2fd;
        padding: 8px;
        border-radius: 6px;
        margin-bottom: 5px;
    }
    .bot-msg {
        background-color: #f5f5f5;
        padding: 8px;
        border-radius: 6px;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Options")
    top_k = st.slider("Top-k documents", 1, 10, 5)
    st.markdown("---")
    st.markdown("""
    <div class="creator-box">
        <strong>👨‍💻 Created by:</strong><br>
        Yoganshu Sharma<br>
        🚀 RAG + Llama-3 + FAISS
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.info("💡 Tip: Press Enter to send. Use Shift+Enter for a new line.")

# Title
st.title("🤖 RAG Chatbot")
st.caption("Ask questions based on your documents")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I'm your RAG-powered chatbot. Ask me anything about your documents!"
    })

# Display history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'>👤 **You:** {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>🤖 **Assistant:** {msg['content']}</div>", unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Type your question here..."):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"<div class='user-msg'>👤 **You:** {prompt}</div>", unsafe_allow_html=True)

    # Placeholder for assistant (typing effect)
    placeholder = st.empty()
    placeholder.markdown("<div class='bot-msg'>🤖 **Assistant:** Thinking...</div>", unsafe_allow_html=True)

    # Measure time
    start_time = time.time()

    # Get response
    try:
        answer = rag_answer(prompt, top_k=top_k)
    except Exception as e:
        answer = f"⚠️ Error: {e}"

    total_time = time.time() - start_time

    # Typing effect
    typed_text = ""
    for char in answer:
        typed_text += char
        placeholder.markdown(f"<div class='bot-msg'>🤖 **Assistant:** {typed_text}▌</div>", unsafe_allow_html=True)
        time.sleep(0.02)  # typing speed

    # Final text (remove cursor ▌)
    placeholder.markdown(f"<div class='bot-msg'>🤖 **Assistant:** {typed_text}</div>", unsafe_allow_html=True)

    # Show response time
    st.caption(f"⏱️ Response generated in {total_time:.2f} seconds")

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})

    st.rerun()
