"""
Streamlit app for a RAG-based chatbot.
Builds the knowledge base from user-uploaded files.
"""

import hashlib
import os
import sys
from typing import Iterable

import streamlit as st

# Add src directory to path
sys.path.append("src")

from rag_pipeline import RAGPipeline


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_VECTOR_STORE = os.getenv("VECTOR_STORE_TYPE", "faiss")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_USE_SIMPLE_LLM = env_flag("USE_SIMPLE_LLM", True)


def initialize_session_state() -> None:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None

    if "pipeline_signature" not in st.session_state:
        st.session_state.pipeline_signature = None

    if "messages" not in st.session_state:
        st.session_state.messages = []


def reset_chat_history() -> None:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Upload files in the sidebar, then ask questions. "
                "I answer using only the uploaded document context."
            ),
            "sources": [],
        }
    ]


def compute_files_signature(uploaded_files: Iterable) -> str:
    digest = hashlib.sha256()

    for uploaded_file in uploaded_files:
        digest.update(uploaded_file.name.encode("utf-8"))
        digest.update(str(uploaded_file.size).encode("utf-8"))
        digest.update(uploaded_file.getvalue())

    return digest.hexdigest()


def normalize_stream_response(stream_response) -> str:
    if isinstance(stream_response, str):
        return stream_response

    if isinstance(stream_response, list):
        return "".join(item for item in stream_response if isinstance(item, str))

    return str(stream_response)


def initialize_pipeline_from_uploads(uploaded_files, use_simple_llm: bool):
    try:
        if use_simple_llm:
            pipeline = RAGPipeline(
                embedding_model=DEFAULT_EMBEDDING_MODEL,
                vector_store_type=DEFAULT_VECTOR_STORE,
                use_simple_llm=True,
            )
        else:
            pipeline = RAGPipeline(
                embedding_model=DEFAULT_EMBEDDING_MODEL,
                llm_model=DEFAULT_OLLAMA_MODEL,
                vector_store_type=DEFAULT_VECTOR_STORE,
                use_simple_llm=False,
                ollama_base_url=DEFAULT_OLLAMA_BASE_URL,
            )

            if not pipeline.llm_handler.check_health():
                st.warning(
                    "Ollama is not reachable. Falling back to simple LLM mode."
                )
                pipeline = RAGPipeline(
                    embedding_model=DEFAULT_EMBEDDING_MODEL,
                    vector_store_type=DEFAULT_VECTOR_STORE,
                    use_simple_llm=True,
                )

        success = pipeline.initialize_from_uploaded_files(uploaded_files)

        if success:
            return pipeline

        st.error("Failed to build knowledge base from uploaded files.")
        return None

    except Exception as error:
        st.error(f"Error initializing pipeline: {error}")
        return None


def display_sources(sources):
    if not sources:
        return

    st.markdown("**Sources**")
    for index, source in enumerate(sources[:3], start=1):
        filename = source.get("filename", "unknown")
        similarity = source.get("similarity_score", 0.0)
        chunk_id = source.get("chunk_id", "n/a")
        word_count = source.get("word_count", "n/a")
        content = source.get("content", "")

        with st.expander(f"Source {index}: {filename} (score: {similarity:.3f})"):
            st.text(content[:500] + "..." if len(content) > 500 else content)
            st.caption(f"Chunk: {chunk_id} | Words: {word_count}")


def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon=":robot_face:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("RAG Chatbot")
    st.caption("Upload one or more files of any type, then ask questions about them.")

    initialize_session_state()
    if not st.session_state.messages:
        reset_chat_history()

    with st.sidebar:
        st.header("Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload files (all file types accepted)",
            accept_multiple_files=True,
        )

        use_simple_llm = st.checkbox(
            "Use simple LLM (recommended for Streamlit Cloud)",
            value=DEFAULT_USE_SIMPLE_LLM,
        )

        st.header("Chat Settings")
        k_docs = st.slider("Documents to retrieve", 1, 10, 5)
        stream_response = st.checkbox("Stream responses", value=True)

        rebuild_clicked = st.button(
            "Build or Refresh Knowledge Base",
            type="primary",
            disabled=not uploaded_files,
        )
        clear_chat_clicked = st.button("Clear Chat", type="secondary")
        reset_pipeline_clicked = st.button("Reset Pipeline", type="secondary")

    if clear_chat_clicked:
        reset_chat_history()
        st.rerun()

    if reset_pipeline_clicked:
        st.session_state.pipeline = None
        st.session_state.pipeline_signature = None
        reset_chat_history()
        st.rerun()

    if not uploaded_files:
        st.info("Upload at least one file in the sidebar to build the knowledge base.")
        st.stop()

    files_signature = compute_files_signature(uploaded_files)
    config_signature = (
        f"{files_signature}|simple={use_simple_llm}|model={DEFAULT_OLLAMA_MODEL}|"
        f"base_url={DEFAULT_OLLAMA_BASE_URL}|embed={DEFAULT_EMBEDDING_MODEL}|"
        f"store={DEFAULT_VECTOR_STORE}"
    )

    needs_rebuild = (
        rebuild_clicked
        or st.session_state.pipeline is None
        or st.session_state.pipeline_signature != config_signature
    )

    if needs_rebuild:
        with st.spinner("Processing uploaded files and building vector index..."):
            pipeline = initialize_pipeline_from_uploads(uploaded_files, use_simple_llm)

        if pipeline is None:
            st.stop()

        st.session_state.pipeline = pipeline
        st.session_state.pipeline_signature = config_signature
        reset_chat_history()
        st.success("Knowledge base is ready.")

    pipeline = st.session_state.pipeline
    if pipeline is None:
        st.error("Pipeline is not initialized.")
        st.stop()

    stats = pipeline.get_stats()
    with st.sidebar:
        st.markdown("---")
        st.subheader("Pipeline Statistics")
        st.write(f"Model: {stats.get('llm_model', 'n/a')}")
        st.write(f"Embeddings: {stats.get('embedding_model', 'n/a')}")
        st.write(f"Vector Store: {str(stats.get('vector_store_type', 'n/a')).upper()}")
        st.write(f"Documents: {stats.get('total_documents', 0)}")
        st.write(f"Chunks: {stats.get('total_chunks', 0)}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            display_sources(message["sources"])

    if prompt := st.chat_input("Ask a question about your uploaded files..."):
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            with st.chat_message("assistant"):
                if stream_response:
                    answer_stream, sources = pipeline.query(prompt, k=k_docs, stream=True)
                    streamed_output = st.write_stream(answer_stream)
                    full_response = normalize_stream_response(streamed_output)
                else:
                    answer, sources = pipeline.query(prompt, k=k_docs, stream=False)
                    st.markdown(answer)
                    full_response = answer

            display_sources(sources)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources,
                }
            )
        except Exception as error:
            st.error(f"Error generating response: {error}")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "I hit an error while processing your question. "
                        "Please try again."
                    ),
                    "sources": [],
                }
            )


if __name__ == "__main__":
    main()
