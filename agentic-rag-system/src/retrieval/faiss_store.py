"""FAISS vector store implementation."""

import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from loguru import logger

from config.settings import settings
from config.constants import FAISS_INDEX_NAME, FAISS_METADATA_INDEX_NAME


class FAISSVectorStore:
    """FAISS-based vector store for document retrieval."""
    
    def __init__(self, embedding_dim: int = 768):
        """Initialize FAISS vector store.
        
        Args:
            embedding_dim: Dimension of embedding vectors
        """
        self.embedding_dim = embedding_dim
        self.index_path = Path(settings.faiss_index_path)
        self.metadata_path = Path(settings.faiss_metadata_path)
        
        # Create directories if they don't exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        self.metadata = []
        
        # Load existing index if available
        self._load_index()
        
        logger.info(f"Initialized FAISS vector store with {self.index.ntotal} vectors")
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Add documents and their embeddings to the store.
        
        Args:
            documents: List of document dictionaries
            embeddings: List of embedding vectors
        """
        try:
            logger.info(f"Adding {len(documents)} documents to FAISS store")
            
            # Convert embeddings to numpy array
            embedding_array = np.array(embeddings).astype('float32')
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embedding_array)
            
            # Add to FAISS index
            self.index.add(embedding_array)
            
            # Store metadata
            for doc in documents:
                self.metadata.append(doc["metadata"])
            
            logger.info(f"Successfully added {len(documents)} documents to FAISS store")
            
        except Exception as e:
            logger.error(f"Error adding documents to FAISS store: {e}")
            raise
    
    def search(self, query_embedding: List[float], k: int = 6) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of similar documents with scores
        """
        try:
            # Convert query to numpy array and normalize
            query_array = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_array)
            
            # Search
            scores, indices = self.index.search(query_array, k)
            
            # Get results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.metadata):
                    result = {
                        "metadata": self.metadata[idx],
                        "score": float(score)
                    }
                    results.append(result)
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error searching FAISS store: {e}")
            raise
    
    def save_index(self):
        """Save the FAISS index and metadata to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            logger.info(f"Saved FAISS index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            raise
    
    def _load_index(self):
        """Load existing FAISS index and metadata from disk."""
        try:
            if self.index_path.exists() and self.metadata_path.exists():
                # Load FAISS index
                self.index = faiss.read_index(str(self.index_path))
                
                # Load metadata
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            else:
                logger.info("No existing FAISS index found, starting fresh")
                
        except Exception as e:
            logger.warning(f"Error loading existing FAISS index: {e}")
            logger.info("Starting with fresh index")
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.metadata = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store.
        
        Returns:
            Dictionary with store statistics
        """
        return {
            "total_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "index_type": "FAISS_IndexFlatIP",
            "metadata_count": len(self.metadata)
        }


def create_faiss_store(embedding_dim: int = 768) -> FAISSVectorStore:
    """Create a FAISS vector store instance.
    
    Args:
        embedding_dim: Dimension of embedding vectors
        
    Returns:
        FAISSVectorStore instance
    """
    return FAISSVectorStore(embedding_dim)