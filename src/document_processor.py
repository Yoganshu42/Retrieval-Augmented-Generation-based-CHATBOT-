"""
Document processing module for RAG chatbot.
Handles document loading, cleaning, and chunking.
"""

import io
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

import PyPDF2
import tiktoken


class DocumentProcessor:
    def __init__(self, chunk_size: int = 200, overlap: int = 50):
        
        # Initialize document processor.
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def load_documents(self, data_dir: str) -> List[Dict[str, str]]:
        
        # Load documents from the data directory.
        
        documents = []
        data_path = Path(data_dir)

        if not data_path.exists():
            return documents

        for file_path in data_path.iterdir():
            if not file_path.is_file():
                continue

            try:
                content = self._extract_content_from_bytes(file_path.read_bytes(), file_path.name)
                if content.strip():
                    documents.append({
                        'content': content,
                        'filename': file_path.name,
                        'filepath': str(file_path),
                        'file_type': file_path.suffix.lower().lstrip('.') or 'unknown'
                    })
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return documents

    def load_uploaded_documents(self, uploaded_files: List[Any]) -> List[Dict[str, str]]:
        """Load documents from Streamlit uploaded files."""
        documents = []

        for uploaded_file in uploaded_files or []:
            filename = getattr(uploaded_file, "name", "uploaded_file")

            try:
                file_bytes = uploaded_file.getvalue()
                content = self._extract_content_from_bytes(file_bytes, filename)

                if not content.strip():
                    continue

                documents.append({
                    'content': content,
                    'filename': filename,
                    'filepath': filename,
                    'file_type': Path(filename).suffix.lower().lstrip('.') or 'unknown'
                })
            except Exception as e:
                print(f"Error loading uploaded file {filename}: {e}")

        return documents

    def _extract_content_from_bytes(self, file_bytes: bytes, filename: str) -> str:
        """Extract readable text from file bytes with broad format support."""
        suffix = Path(filename).suffix.lower()

        if suffix == ".pdf":
            return self._extract_pdf_bytes_content(file_bytes)

        if suffix == ".docx":
            return self._extract_docx_content(file_bytes)

        if suffix == ".json":
            raw = self._decode_bytes(file_bytes)
            if not raw.strip():
                return ""
            try:
                parsed = json.loads(raw)
                return json.dumps(parsed, indent=2, ensure_ascii=True)
            except json.JSONDecodeError:
                return raw

        if suffix in {".html", ".htm"}:
            return self._strip_html(self._decode_bytes(file_bytes))

        decoded = self._decode_bytes(file_bytes)
        if decoded.strip():
            return decoded

        # Keep a lightweight textual placeholder for binary formats so all file
        # types can still be indexed and cited in responses.
        return (
            f"File '{filename}' was uploaded, but it appears to be a binary format. "
            "Text extraction is limited for this file type."
        )
    
    def _extract_pdf_content(self, file_path: Path) -> str:
        try:
            with open(file_path, 'rb') as f:
                return self._extract_pdf_bytes_content(f.read())
        except Exception as e:
            print(f"Error extracting PDF content from {file_path}: {e}")

        return ""

    def _extract_pdf_bytes_content(self, file_bytes: bytes) -> str:
        content_parts: List[str] = []

        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    content_parts.append(page_text)
        except Exception as e:
            print(f"Error extracting PDF bytes content: {e}")

        return "\n".join(content_parts)

    def _extract_docx_content(self, file_bytes: bytes) -> str:
        text_segments: List[str] = []

        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
                if "word/document.xml" not in archive.namelist():
                    return ""

                xml_content = archive.read("word/document.xml")

            root = ET.fromstring(xml_content)
            namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

            for node in root.iter(f"{namespace}t"):
                if node.text:
                    text_segments.append(node.text)
        except Exception as e:
            print(f"Error extracting DOCX content: {e}")
            return ""

        return " ".join(text_segments)

    def _decode_bytes(self, file_bytes: bytes) -> str:
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                text = file_bytes.decode(encoding)
                if self._looks_like_text(text):
                    return text
            except UnicodeDecodeError:
                continue

        return ""

    def _looks_like_text(self, text: str) -> bool:
        cleaned = text.strip()
        if not cleaned:
            return False

        printable_chars = sum(char.isprintable() or char in "\r\n\t" for char in cleaned)
        return (printable_chars / len(cleaned)) >= 0.85

    def _strip_html(self, text: str) -> str:
        without_scripts = re.sub(
            r"<(script|style)[^>]*>.*?</\1>",
            " ",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
        without_tags = re.sub(r"<[^>]+>", " ", without_scripts)
        return re.sub(r"\s+", " ", without_tags).strip()
    
    def clean_text(self, text: str) -> str:
        
        # Clean text content.
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters 
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Stripping whitespace
        text = text.strip()
        
        return text
    
    def sentence_aware_split(self, text: str, max_words: int) -> List[str]:
        
        # Split text into chunks while preserving sentence boundaries.
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_word_count + sentence_words <= max_words:
                current_chunk.append(sentence)
                current_word_count += sentence_words
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_word_count = sentence_words
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def create_overlapping_chunks(self, chunks: List[str]) -> List[str]:
        
        # Add overlap between consecutive chunks.
      
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_words = chunks[i-1].split()
            curr_words = chunks[i].split()
            
            # Take last words from previous chunk
            overlap_words = prev_words[-self.overlap:] if len(prev_words) > self.overlap else prev_words
            
            # Combine with current chunk
            overlapped_chunk = ' '.join(overlap_words + curr_words)
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        
        all_chunks = []
        
        for doc in documents:
            content = self.clean_text(doc['content'])
            
            # Split into initial chunks
            chunks = self.sentence_aware_split(content, self.chunk_size)
            
            # Add overlap
            overlapped_chunks = self.create_overlapping_chunks(chunks)
            
            # Create chunk objects 
            for i, chunk in enumerate(overlapped_chunks):
                chunk_obj = {
                    'content': chunk,
                    'filename': doc['filename'],
                    'filepath': doc['filepath'],
                    'chunk_id': f"{doc['filename']}_chunk_{i}",
                    'chunk_index': i,
                    'word_count': len(chunk.split())
                }
                all_chunks.append(chunk_obj)
        
        return all_chunks
    
    def save_chunks(self, chunks: List[Dict[str, str]], output_dir: str):
                
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save individual chunk files
        for chunk in chunks:
            chunk_file = output_path / f"{chunk['chunk_id']}.txt"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(chunk['content'])
        
        # Save metadata
        metadata_file = output_path / "chunks_metadata.txt"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(f"ID: {chunk['chunk_id']}\n")
                f.write(f"File: {chunk['filename']}\n")
                f.write(f"Words: {chunk['word_count']}\n")
                f.write(f"Content: {chunk['content'][:100]}...\n")
                f.write("-" * 50 + "\n")


def main():
    # Test the document processor
    processor = DocumentProcessor(chunk_size=200, overlap=50)
    
    # Load documents
    documents = processor.load_documents("data")
    print(f"Loaded {len(documents)} documents")
    
    # Chunk documents
    chunks = processor.chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Save chunks
    processor.save_chunks(chunks, "chunks")
    print("Chunks saved successfully")


if __name__ == "__main__":
    main()
