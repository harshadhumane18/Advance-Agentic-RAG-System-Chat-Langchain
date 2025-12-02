"""Google Gemini embeddings implementation."""

import os
from typing import List, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings
from loguru import logger

from config.settings import settings
from config.constants import FAISS_INDEX_NAME, FAISS_METADATA_INDEX_NAME

class GeminiEmbeddings(Embeddings):
    """Google Gemini embeddings wrapper."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize Gemini embeddings.
        
        Args:
            model_name: Name of the Gemini embedding model to use
        """
        self.model_name = model_name or settings.embedding_model
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=self.model_name,
            google_api_key=settings.google_api_key
        )
        logger.info(f"Initialized Gemini embeddings with model: {self.model_name}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            logger.info(f"Embedding {len(texts)} documents")
            embeddings = self._embeddings.embed_documents(texts)
            logger.info(f"Successfully embedded {len(embeddings)} documents")
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        try:
            logger.info(f"Embedding query: {text[:100]}...")
            embedding = self._embeddings.embed_query(text)
            logger.info("Successfully embedded query")
            return embedding
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise


def get_embeddings_model(model_name: Optional[str] = None) -> GeminiEmbeddings:
    """Get a Gemini embeddings model instance.
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        GeminiEmbeddings instance
    """
    return GeminiEmbeddings(model_name)