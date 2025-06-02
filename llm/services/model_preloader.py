"""
Model preloading service to eliminate cold start delays
Warms up LLM models and embedding models on startup
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

import openai

logger = logging.getLogger(__name__)


@dataclass
class PreloadConfig:
    """Configuration for model preloading"""
    
    # LLM preloading
    enable_llm_preload: bool = True
    preload_extraction_model: bool = True
    preload_formatting_model: bool = True
    preload_timeout_seconds: int = 60
    
    # Embedding preloading  
    enable_embedding_preload: bool = True
    embedding_preload_timeout_seconds: int = 30
    
    # Preload queries (lightweight test queries)
    llm_warmup_queries: list = None
    embedding_warmup_texts: list = None
    
    def __post_init__(self):
        if self.llm_warmup_queries is None:
            self.llm_warmup_queries = [
                "What is the main phone number?",
                "How do I handle customer concerns?",
                "Test warmup query"
            ]
        
        if self.embedding_warmup_texts is None:
            self.embedding_warmup_texts = [
                "phone number contact information",
                "customer service workflow",
                "warmup embedding test"
            ]


class ModelPreloader:
    """Handles preloading of LLM and embedding models to eliminate cold starts"""
    
    def __init__(self, llm_service=None, embedding_service=None, config: Optional[PreloadConfig] = None):
        """
        Initialize model preloader
        
        Args:
            llm_service: LLM service instance
            embedding_service: Embedding service instance  
            config: Preloading configuration
        """
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.config = config or PreloadConfig()
        
        # Preloading state
        self.llm_preloaded = False
        self.embedding_preloaded = False
        self.preload_start_time = None
        self.preload_results = {
            "llm_extraction_preload": {"success": False, "time_seconds": 0},
            "llm_formatting_preload": {"success": False, "time_seconds": 0}, 
            "embedding_preload": {"success": False, "time_seconds": 0},
            "total_preload_time": 0
        }
    
    async def preload_all_models(self) -> Dict[str, Any]:
        """
        Preload all configured models
        
        Returns:
            Dict with preloading results and timing
        """
        logger.info("🚀 Starting model preloading to eliminate cold starts...")
        self.preload_start_time = time.time()
        
        # Run preloading tasks concurrently
        tasks = []
        
        if self.config.enable_llm_preload and self.llm_service:
            if self.config.preload_extraction_model:
                tasks.append(self._preload_llm_extraction())
            if self.config.preload_formatting_model:
                tasks.append(self._preload_llm_formatting())
        
        if self.config.enable_embedding_preload and self.embedding_service:
            tasks.append(self._preload_embeddings())
        
        # Execute all preloading tasks concurrently
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error during concurrent model preloading: {e}")
        
        # Clean up any async resources created during preloading
        await self._cleanup_preload_resources()
        
        # Calculate total time
        total_time = time.time() - self.preload_start_time
        self.preload_results["total_preload_time"] = round(total_time, 2)
        
        # Update preloading status
        self.llm_preloaded = (
            self.preload_results["llm_extraction_preload"]["success"] and 
            self.preload_results["llm_formatting_preload"]["success"]
        )
        self.embedding_preloaded = self.preload_results["embedding_preload"]["success"]
        
        logger.info(f"✅ Model preloading completed in {total_time:.1f}s")
        logger.info(f"   LLM Extraction: {self.preload_results['llm_extraction_preload']['success']} ({self.preload_results['llm_extraction_preload']['time_seconds']}s)")
        logger.info(f"   LLM Formatting: {self.preload_results['llm_formatting_preload']['success']} ({self.preload_results['llm_formatting_preload']['time_seconds']}s)")  
        logger.info(f"   Embeddings: {self.preload_results['embedding_preload']['success']} ({self.preload_results['embedding_preload']['time_seconds']}s)")
        
        return self.preload_results
    
    async def _preload_llm_extraction(self) -> None:
        """Preload extraction model with warmup query using pure sync operations"""
        start_time = time.time()
        
        try:
            logger.info("🔥 Preloading LLM extraction model...")
            
            # Import here to avoid circular imports
            from core.config import model_config
            
            # Use first warmup query
            warmup_query = self.config.llm_warmup_queries[0]
            warmup_context = "This is a warmup context to preload the extraction model."
            
            # Create simple extraction prompt
            prompt = f"Extract the answer to this question from the context.\nQuestion: {warmup_query}\nContext: {warmup_context}\nAnswer: Please provide a brief response."
            
            # Use pure sync OpenAI client to avoid async lifecycle issues
            response = self._make_sync_llm_call(
                prompt=prompt,
                model=model_config.extraction_model,
                max_tokens=50,  # Minimal tokens for warmup
                temperature=0.0
            )
            
            if "error" not in response:
                preload_time = time.time() - start_time
                self.preload_results["llm_extraction_preload"] = {
                    "success": True, 
                    "time_seconds": round(preload_time, 2),
                    "response_length": len(response.get("content", ""))
                }
                logger.info(f"✅ LLM extraction model preloaded in {preload_time:.1f}s")
            else:
                raise Exception(f"Extraction warmup failed: {response.get('error')}")
                
        except Exception as e:
            preload_time = time.time() - start_time
            logger.error(f"❌ LLM extraction preloading failed: {e}")
            self.preload_results["llm_extraction_preload"] = {
                "success": False,
                "time_seconds": round(preload_time, 2), 
                "error": str(e)
            }
    
    async def _preload_llm_formatting(self) -> None:
        """Preload formatting model with warmup query using pure sync operations"""
        start_time = time.time()
        
        try:
            logger.info("🔥 Preloading LLM formatting model...")
            
            # Import here to avoid circular imports
            from core.config import model_config
            
            # Simple formatting prompt
            warmup_text = "The main phone number is 480-999-3046."
            prompt = f"Format this text for presentation: {warmup_text}"
            
            # Use pure sync OpenAI client to avoid async lifecycle issues
            response = self._make_sync_llm_call(
                prompt=prompt,
                model=model_config.formatting_model,
                max_tokens=50,  # Minimal tokens for warmup
                temperature=0.0
            )
            
            if "error" not in response:
                preload_time = time.time() - start_time
                self.preload_results["llm_formatting_preload"] = {
                    "success": True,
                    "time_seconds": round(preload_time, 2),
                    "response_length": len(response.get("content", ""))
                }
                logger.info(f"✅ LLM formatting model preloaded in {preload_time:.1f}s")
            else:
                raise Exception(f"Formatting warmup failed: {response.get('error')}")
                
        except Exception as e:
            preload_time = time.time() - start_time
            logger.error(f"❌ LLM formatting preloading failed: {e}")
            self.preload_results["llm_formatting_preload"] = {
                "success": False,
                "time_seconds": round(preload_time, 2),
                "error": str(e)
            }
    
    async def _preload_embeddings(self) -> None:
        """Preload embedding model with warmup texts"""
        start_time = time.time()
        
        try:
            logger.info("🔥 Preloading embedding model...")
            
            # Use embedding service to generate embeddings for warmup texts
            warmup_texts = self.config.embedding_warmup_texts
            
            # Create embeddings for warmup (this loads the model)
            if hasattr(self.embedding_service, 'embed_documents'):
                await asyncio.wait_for(
                    self.embedding_service.embed_documents(warmup_texts),
                    timeout=self.config.embedding_preload_timeout_seconds
                )
            else:
                # Fallback if async method not available
                import asyncio
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, 
                    self.embedding_service.embed_documents,
                    warmup_texts
                )
            
            preload_time = time.time() - start_time
            self.preload_results["embedding_preload"] = {
                "success": True,
                "time_seconds": round(preload_time, 2),
                "texts_processed": len(warmup_texts)
            }
            logger.info(f"✅ Embedding model preloaded in {preload_time:.1f}s")
            
        except Exception as e:
            preload_time = time.time() - start_time
            logger.error(f"❌ Embedding preloading failed: {e}")
            self.preload_results["embedding_preload"] = {
                "success": False,
                "time_seconds": round(preload_time, 2),
                "error": str(e)
            }
    
    def is_preloaded(self) -> bool:
        """Check if models are preloaded"""
        return self.llm_preloaded and self.embedding_preloaded
    
    def get_preload_status(self) -> Dict[str, Any]:
        """Get detailed preloading status"""
        return {
            "llm_preloaded": self.llm_preloaded,
            "embedding_preloaded": self.embedding_preloaded,
            "is_fully_preloaded": self.is_preloaded(),
            "preload_results": self.preload_results
        }
    
    async def ensure_preloaded(self) -> bool:
        """Ensure models are preloaded, run preloading if needed"""
        if not self.is_preloaded():
            logger.info("Models not preloaded, running preload now...")
            await self.preload_all_models()
        return self.is_preloaded()
    
    async def _cleanup_preload_resources(self) -> None:
        """Clean up any async resources created during preloading"""
        try:
            # If the LLM service has async cleanup methods, call them
            if hasattr(self.llm_service, 'close_connections'):
                await self.llm_service.close_connections()
                logger.debug("Cleaned up LLM service connections after preloading")
        except Exception as e:
            logger.debug(f"Note: Error during preload cleanup (non-critical): {e}")
    
    def _make_sync_llm_call(self, prompt: str, model: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """Make a synchronous LLM call using a separate OpenAI client"""
        try:
            # Create a separate sync OpenAI client for preloading to avoid async issues
            sync_client = openai.OpenAI(api_key=self.llm_service.api_key)
            
            # Make the API call
            response = sync_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate and concise answers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=30.0
            )
            
            # Format response
            usage = response.usage if hasattr(response, 'usage') else {}
            return {
                "content": (response.choices[0].message.content or "").strip(),
                "model": response.model,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                },
                "id": response.id,
                "created": response.created
            }
            
        except Exception as e:
            logger.error(f"Sync LLM call failed: {e}")
            return {
                "error": str(e),
                "content": f"Error: {e}",
                "usage": {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0},
                "model": "error"
            }
        finally:
            # Ensure the sync client is properly closed
            try:
                if 'sync_client' in locals():
                    sync_client.close()
            except:
                pass  # Ignore cleanup errors
    
    async def cleanup(self) -> None:
        """Public cleanup method for external use"""
        await self._cleanup_preload_resources()


# Global preloader instance (will be initialized by application)
global_preloader: Optional[ModelPreloader] = None


def get_global_preloader() -> Optional[ModelPreloader]:
    """Get the global model preloader instance"""
    return global_preloader


def set_global_preloader(preloader: ModelPreloader) -> None:
    """Set the global model preloader instance"""
    global global_preloader
    global_preloader = preloader


async def preload_models_if_needed(llm_service=None, embedding_service=None) -> Dict[str, Any]:
    """
    Convenience function to preload models if not already done
    
    Args:
        llm_service: LLM service instance
        embedding_service: Embedding service instance
        
    Returns:
        Preloading results
    """
    global global_preloader
    
    if global_preloader is None:
        config = PreloadConfig()
        global_preloader = ModelPreloader(llm_service, embedding_service, config)
    
    if not global_preloader.is_preloaded():
        return await global_preloader.preload_all_models()
    else:
        logger.info("Models already preloaded, skipping...")
        return global_preloader.get_preload_status()
