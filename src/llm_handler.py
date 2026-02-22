"""
LLM handler module for RAG chatbot.
Handles language model initialization and response generation using Ollama.
"""

import re
from typing import Dict, Iterator, List
import requests

try:
    import ollama
except ImportError:  # pragma: no cover - environment dependent
    ollama = None


class LLMHandler:
    def __init__(self, model_name: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        
        # Initialize Ollama LLM handler.
        if ollama is None:
            raise RuntimeError(
                "Ollama Python package is not installed. "
                "Install 'ollama' or run with use_simple_llm=True."
            )

        self.model_name = model_name
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)
        
        # Using Ollama's built-in model
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

    def _extract_model_name(self, model: Dict | object) -> str:
        """Extract model name from Ollama list response across SDK versions."""
        if isinstance(model, dict):
            return model.get("model") or model.get("name") or ""
        return getattr(model, "model", "") or getattr(model, "name", "")
    
    def ensure_model_available(self):
        """Ensure the specified model is available locally."""
        try:
            print(f"Checking if model {self.model_name} is available...")
            
            # List available models
            models = self.client.list()
            model_items = models.get("models", []) if isinstance(models, dict) else getattr(models, "models", [])
            available_model_names = [self._extract_model_name(model) for model in model_items]
            
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
        
        # Create a RAG prompt with retrieved context.
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            context_parts.append(f"Source {i+1}: {chunk['content']}")
        
        context = "\n\n".join(context_parts)
        
        # prompt template
        prompt = f"""Answer only from the provided context. If the answer is not explicitly present, say that it is not available in the uploaded files.
Keep the answer concise and do not invent facts.

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
        
        # Generate a response from the Ollama model.        
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
        
        # Generate a streaming response from the Ollama model.
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
        
        # Answer a query using RAG approach with Ollama.
        prompt = self.create_rag_prompt(query, retrieved_chunks)
        
        if stream:
            return self.generate_streaming_response(prompt)
        else:
            return self.generate_response(prompt)
    
    def list_available_models(self) -> List[str]:
        
        # List all available models in the Ollama instance.
        try:
            models = self.client.list()
            model_items = models.get("models", []) if isinstance(models, dict) else getattr(models, "models", [])
            return [self._extract_model_name(model) for model in model_items if self._extract_model_name(model)]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def check_health(self) -> bool:
        # Check if Ollama server is healthy and responsive.
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False


# Fallback simple LLM for testing
class SimpleLLM:
    # Simple rule-based LLM for testing when GPU models aren't available
    
    def __init__(self):
        self.model_name = "Simple Rule-based Model"
        self.term_aliases = {
            "ml": ["ml", "machine learning"],
            "ai": ["ai", "artificial intelligence"],
            "dl": ["dl", "deep learning"],
            "ds": ["ds", "data science"],
            "nlp": ["nlp", "natural language processing"],
        }

    def _query_terms(self, query: str) -> List[str]:
        lowered = query.lower()
        raw_tokens = re.findall(r"[a-zA-Z0-9]+", lowered)
        terms = [token for token in raw_tokens if len(token) > 1]

        expanded_terms: List[str] = []
        for term in terms:
            expanded_terms.append(term)
            expanded_terms.extend(self.term_aliases.get(term, []))

        # Keep order stable while removing duplicates.
        return list(dict.fromkeys(expanded_terms))

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text or "")
        return [sentence.strip() for sentence in sentences if sentence and sentence.strip()]

    def _term_match_count(self, sentence: str, terms: List[str]) -> int:
        lowered_sentence = sentence.lower()
        count = 0
        for term in terms:
            if " " in term:
                if term in lowered_sentence:
                    count += 1
            else:
                if re.search(rf"\b{re.escape(term)}\b", lowered_sentence):
                    count += 1
        return count

    def _sentence_score(self, sentence: str, query: str, terms: List[str], similarity_score: float) -> float:
        lowered_sentence = sentence.lower()
        score = max(similarity_score, 0.0) * 2.0

        term_hits = self._term_match_count(sentence, terms)
        score += float(term_hits)

        for term in terms:
            if " " in term and term in lowered_sentence:
                score += 2.0

        # Boost definition-like statements for "what is/define" style questions.
        lowered_query = query.lower()
        definition_query = any(phrase in lowered_query for phrase in ["what is", "define", "meaning of"])
        if definition_query:
            for term in terms:
                if (
                    f"{term} is" in lowered_sentence
                    or f"{term} stands for" in lowered_sentence
                    or f"{term} refers to" in lowered_sentence
                ):
                    score += 2.0

        return score

    def _extractive_answer(self, query: str, retrieved_chunks: List[Dict[str, str]]) -> str:
        if not retrieved_chunks:
            return "I could not find relevant information in the uploaded files."

        terms = self._query_terms(query)
        candidates: List[tuple[float, int, str, str]] = []
        lowered_query = query.lower()
        definition_query = any(phrase in lowered_query for phrase in ["what is", "define", "meaning of"])

        for chunk in retrieved_chunks:
            similarity_score = float(chunk.get("similarity_score", 0.0))
            sentences = self._split_sentences(chunk.get("content", ""))
            filename = chunk.get("filename", "unknown source")

            for sentence in sentences:
                score = self._sentence_score(sentence, query, terms, similarity_score)
                term_hits = self._term_match_count(sentence, terms)
                if score > 0:
                    candidates.append((score, term_hits, sentence, filename))

        if not candidates:
            return "I could not find relevant information in the uploaded files."

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        top_score = candidates[0][0]
        if top_score < 2.5:
            return (
                "I could not find a direct answer in the uploaded files for that question. "
                "Try asking with more context."
            )

        if definition_query and candidates[0][1] == 0:
            return (
                "I could not find a direct definition in the uploaded files for that question. "
                "Try asking with the full term name."
            )

        selected: List[tuple[str, str]] = []
        used_sentences = set()
        for score, term_hits, sentence, filename in candidates:
            normalized = sentence.strip().lower()
            if normalized in used_sentences:
                continue

            if selected and (term_hits == 0 or score < max(2.5, top_score * 0.6)):
                continue

            selected.append((sentence.strip(), filename))
            used_sentences.add(normalized)
            if len(selected) >= 2:
                break

        if not selected:
            return "I could not find relevant information in the uploaded files."

        answer_lines = [f"Based on the uploaded files: {selected[0][0]}"]
        if len(selected) > 1:
            answer_lines.append(f"Additional context: {selected[1][0]}")

        source_names = ", ".join(dict.fromkeys(filename for _, filename in selected))
        answer_lines.append(f"Source: {source_names}")
        return "\n\n".join(answer_lines)
    
    def answer_query(self, query: str, retrieved_chunks: List[Dict[str, str]], stream: bool = False):
        #Generate a simple response based on retrieved chunks
        response = self._extractive_answer(query, retrieved_chunks)
        
        if stream:
            def response_stream() -> Iterator[str]:
                # Simulate streaming by yielding words
                for word in response.split():
                    yield word + " "
            return response_stream()

        return response


def main():
    # Test the LLM handler
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
    
    # Test simple LLM
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
