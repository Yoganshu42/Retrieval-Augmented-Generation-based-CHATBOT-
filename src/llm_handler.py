"""
LLM handler module for RAG chatbot.
Handles language model initialization and response generation using Ollama.
"""

import os
from typing import List, Dict, Iterator, Optional
import ollama
import requests
import json
import time


class LLMHandler:
    def __init__(self, model_name: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama LLM handler.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL of the Ollama server
        """
        self.model_name = model_name
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)
        
        # Available models - these are popular Ollama models
        self.available_models = {
            "llama3.2:3b": "llama3.2:3b",
            "llama3.2:1b": "llama3.2:1b", 
            "llama3.1:8b": "llama3.1:8b",
            "mistral:7b": "mistral:7b",
            "codellama:7b": "codellama:7b",
            "phi3:mini": "phi3:mini",
            "gemma2:2b": "gemma2:2b"
        }
        
        # Ensure model is available
        self.ensure_model_available()
    
    def ensure_model_available(self):
        """Ensure the specified model is available locally."""
        try:
            print(f"Checking if model {self.model_name} is available...")
            
            # List available models
            models = self.client.list()
            available_model_names = [model['model'] for model in models['models']]
            
            if self.model_name not in available_model_names:
                print(f"Model {self.model_name} not found locally. Pulling from Ollama registry...")
                print("This may take a few minutes depending on the model size...")
                
                # Pull the model
                self.client.pull(self.model_name)
                print(f"Successfully pulled model: {self.model_name}")
            else:
                print(f"Model {self.model_name} is already available.")
                
        except Exception as e:
            print(f"Error checking/pulling model: {e}")
            print("Make sure Ollama is running and accessible.")
            print(f"You can start Ollama with 'ollama serve' or check if it's running at {self.base_url}")
            raise
    
    def create_rag_prompt(self, query: str, retrieved_chunks: List[Dict[str, str]]) -> str:
        """
        Create a RAG prompt with retrieved context.
        
        Args:
            query: User query
            retrieved_chunks: List of retrieved document chunks
            
        Returns:
            Formatted prompt string
        """
        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            context_parts.append(f"Source {i+1}: {chunk['content']}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt template
        prompt = f"""Based on the following context, please provide a helpful and accurate answer to the user's question. If the answer cannot be found in the context, please say so.

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def generate_response(
        self, 
        prompt: str, 
        max_tokens: int = 512, 
        temperature: float = 0.7
    ) -> str:
        """
        Generate a response from the Ollama model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                    'top_p': 0.9,
                    'repeat_penalty': 1.1
                }
            )
            
            return response['response'].strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while generating a response. Please make sure Ollama is running."
    
    def generate_streaming_response(
        self, 
        prompt: str, 
        max_tokens: int = 512, 
        temperature: float = 0.7
    ) -> Iterator[str]:
        """
        Generate a streaming response from the Ollama model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate 
            temperature: Sampling temperature
            
        Yields:
            Generated tokens as they are produced
        """
        try:
            stream = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=True,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                    'top_p': 0.9,
                    'repeat_penalty': 1.1
                }
            )
            
            for chunk in stream:
                if 'response' in chunk:
                    yield chunk['response']
                    
        except Exception as e:
            print(f"Error generating streaming response: {e}")
            yield "I'm sorry, I encountered an error while generating a response. Please make sure Ollama is running."
    
    def answer_query(
        self, 
        query: str, 
        retrieved_chunks: List[Dict[str, str]], 
        stream: bool = False
    ) -> str | Iterator[str]:
        """
        Answer a query using RAG approach with Ollama.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved document chunks
            stream: Whether to stream the response
            
        Returns:
            Generated answer (string or iterator)
        """
        prompt = self.create_rag_prompt(query, retrieved_chunks)
        
        if stream:
            return self.generate_streaming_response(prompt)
        else:
            return self.generate_response(prompt)
    
    def list_available_models(self) -> List[str]:
        """
        List all available models in the Ollama instance.
        
        Returns:
            List of available model names
        """
        try:
            models = self.client.list()
            return [model['name'] for model in models['models']]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def check_health(self) -> bool:
        """
        Check if Ollama server is healthy and responsive.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False


# Fallback simple LLM for testing
class SimpleLLM:
    """Simple rule-based LLM for testing when GPU models aren't available."""
    
    def __init__(self):
        self.model_name = "Simple Rule-based Model"
    
    def answer_query(self, query: str, retrieved_chunks: List[Dict[str, str]], stream: bool = False):
        """Generate a simple response based on retrieved chunks."""
        if not retrieved_chunks:
            response = "I don't have enough information to answer your question."
        else:
            # Create a simple response from the most relevant chunk
            best_chunk = retrieved_chunks[0]
            relevant_content = best_chunk['content'][:300] + "..."
            
            response = f"Based on the available information: {relevant_content}"
            
            # Add source information
            response += f"\n\nSource: {best_chunk['filename']}"
        
        if stream:
            # Simulate streaming by yielding words
            words = response.split()
            for word in words:
                yield word + " "
        else:
            return response


def main():
    """Test the LLM handler."""
    # Test Ollama LLM first
    print("Testing Ollama LLM...")
    try:
        ollama_llm = LLMHandler()
        
        # Check health first
        if not ollama_llm.check_health():
            print("Ollama server is not running. Please start it with 'ollama serve'")
            return
        
        print("Available models:", ollama_llm.list_available_models())
        
        mock_chunks = [
            {
                'content': 'Our privacy policy protects user data and ensures confidentiality.',
                'filename': 'terms.txt',
                'similarity_score': 0.95
            }
        ]
        
        print("\nTesting non-streaming response:")
        response = ollama_llm.answer_query("What is your privacy policy?", mock_chunks)
        print("Response:", response)
        
        print("\nTesting streaming response:")
        for token in ollama_llm.answer_query("What is your privacy policy?", mock_chunks, stream=True):
            print(token, end="", flush=True)
        print()
        
    except Exception as e:
        print(f"Error testing Ollama LLM: {e}")
    
    # Test simple LLM as fallback
    print("\nTesting Simple LLM (fallback)...")
    simple_llm = SimpleLLM()
    
    mock_chunks = [
        {
            'content': 'Our privacy policy protects user data and ensures confidentiality.',
            'filename': 'terms.txt',
            'similarity_score': 0.95
        }
    ]
    
    response = simple_llm.answer_query("What is your privacy policy?", mock_chunks)
    print("Response:", response)


if __name__ == "__main__":
    main()