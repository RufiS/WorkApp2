"""
Ollama LLM Service for cost-effective local answer generation.
Connects to external Ollama server running qwen2.5:14b-instruct.
"""

import json
import time
import logging
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OllamaConfig:
    """Configuration for Ollama service."""
    base_url: str = "http://192.168.254.204:11434"
    model: str = "qwen2.5:14b-instruct"
    timeout: int = 60
    max_tokens: int = 4096
    temperature: float = 0.0
    
class OllamaLLMService:
    """Local LLM service using Ollama server."""
    
    def __init__(self, config: OllamaConfig = None):
        """Initialize Ollama LLM service."""
        self.config = config or OllamaConfig()
        self.session = requests.Session()
        logger.info(f"Initialized Ollama service: {self.config.base_url} with model {self.config.model}")
        
        # Test connection
        if not self._test_connection():
            raise ConnectionError(f"Cannot connect to Ollama server at {self.config.base_url}")
    
    def _test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            response = self.session.get(f"{self.config.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                if any(self.config.model in name for name in model_names):
                    logger.info(f"✅ Ollama connection successful, model {self.config.model} available")
                    return True
                else:
                    logger.warning(f"⚠️ Model {self.config.model} not found. Available: {model_names}")
                    return False
            return False
        except Exception as e:
            logger.error(f"❌ Ollama connection failed: {e}")
            return False
    
    def generate_answer(self, question: str, context: str) -> Dict[str, Any]:
        """
        Generate answer using Ollama model.
        
        Args:
            question: The question to answer
            context: Retrieved context from documents
            
        Returns:
            Dict with 'content' key containing the answer
        """
        start_time = time.time()
        
        try:
            # Construct prompt
            prompt = self._build_prompt(question, context)
            
            # Call Ollama API
            payload = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                    "stop": ["Human:", "Assistant:", "\n\nHuman:", "\n\nAssistant:"]
                }
            }
            
            logger.debug(f"🤖 Ollama request: {self.config.model} with {len(prompt)} chars")
            
            response = self.session.post(
                f"{self.config.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
            
            result = response.json()
            answer = result.get('response', '').strip()
            
            generation_time = time.time() - start_time
            
            logger.info(f"✅ Ollama generation: {generation_time:.2f}s, {len(answer)} chars")
            
            return {
                'content': answer,
                'model': self.config.model,
                'generation_time': generation_time,
                'prompt_tokens': len(prompt.split()),
                'completion_tokens': len(answer.split()),
                'cost': 0.0  # Local model is free
            }
            
        except Exception as e:
            logger.error(f"❌ Ollama generation failed: {e}")
            return {
                'content': f"ERROR: Ollama generation failed - {str(e)}",
                'model': self.config.model,
                'generation_time': time.time() - start_time,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'cost': 0.0,
                'error': str(e)
            }
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build prompt for answer generation."""
        
        system_prompt = """You are an expert assistant for a dispatch and customer service system. Your job is to provide accurate, helpful answers based on the provided context.

Instructions:
- Answer based ONLY on the provided context
- Be specific and actionable 
- If the context doesn't contain enough information, say so
- For procedures, list steps clearly
- Include relevant phone numbers, codes, or specifics when available
- Keep answers concise but complete"""

        user_prompt = f"""Context from company documents:
{context}

Question: {question}

Answer:"""

        return f"{system_prompt}\n\n{user_prompt}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        try:
            response = self.session.get(f"{self.config.base_url}/api/show", 
                                      json={"name": self.config.model},
                                      timeout=10)
            if response.status_code == 200:
                return response.json()
            return {"error": f"Cannot get model info: {response.status_code}"}
        except Exception as e:
            return {"error": f"Cannot get model info: {e}"}
    
    def is_available(self) -> bool:
        """Check if the service is available."""
        return self._test_connection()
