"""
Document processing module for RAG chatbot.
Handles document loading, cleaning, and chunking.
"""

import os
import re
from typing import List, Dict
from pathlib import Path
import tiktoken
import PyPDF2
import io


class DocumentProcessor:
    def __init__(self, chunk_size: int = 200, overlap: int = 50):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Target number of words per chunk
            overlap: Number of words to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def load_documents(self, data_dir: str) -> List[Dict[str, str]]:
        """
        Load all documents from the data directory.
        Supports both .txt and .pdf files.
        
        Args:
            data_dir: Path to directory containing documents
            
        Returns:
            List of documents with metadata
        """
        documents = []
        data_path = Path(data_dir)
        
        # Load .txt files
        for file_path in data_path.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append({
                        'content': content,
                        'filename': file_path.name,
                        'filepath': str(file_path),
                        'file_type': 'txt'
                    })
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        # Load .pdf files
        for file_path in data_path.glob("*.pdf"):
            try:
                content = self._extract_pdf_content(file_path)
                if content.strip():  # Only add if content is not empty
                    documents.append({
                        'content': content,
                        'filename': file_path.name,
                        'filepath': str(file_path),
                        'file_type': 'pdf'
                    })
            except Exception as e:
                print(f"Error loading PDF {file_path}: {e}")
        
        return documents
    
    def _extract_pdf_content(self, file_path: Path) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        content = ""
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    content += page_text + "\n"
                    
        except Exception as e:
            print(f"Error extracting PDF content from {file_path}: {e}")
            
        return content
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def sentence_aware_split(self, text: str, max_words: int) -> List[str]:
        """
        Split text into chunks while preserving sentence boundaries.
        
        Args:
            text: Text to split
            max_words: Maximum words per chunk
            
        Returns:
            List of text chunks
        """
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
        """
        Add overlap between consecutive chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of overlapping chunks
        """
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_words = chunks[i-1].split()
            curr_words = chunks[i].split()
            
            # Take last 'overlap' words from previous chunk
            overlap_words = prev_words[-self.overlap:] if len(prev_words) > self.overlap else prev_words
            
            # Combine with current chunk
            overlapped_chunk = ' '.join(overlap_words + curr_words)
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Chunk documents into smaller segments.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of document chunks with metadata
        """
        all_chunks = []
        
        for doc in documents:
            content = self.clean_text(doc['content'])
            
            # Split into initial chunks
            chunks = self.sentence_aware_split(content, self.chunk_size)
            
            # Add overlap
            overlapped_chunks = self.create_overlapping_chunks(chunks)
            
            # Create chunk objects with metadata
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
        """
        Save processed chunks to files.
        
        Args:
            chunks: List of chunks to save
            output_dir: Directory to save chunks
        """
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
    """Test the document processor."""
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