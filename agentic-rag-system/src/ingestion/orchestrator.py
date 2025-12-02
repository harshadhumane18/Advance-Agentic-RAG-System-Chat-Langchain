"""Main ingestion orchestrator."""

from typing import List, Dict, Any
from loguru import logger
from tqdm import tqdm

from config.settings import settings
from embeddings.gemini_embeddings import get_embeddings_model
from retrieval.faiss_store import create_faiss_store
from ingestion.document_loaders import create_document_loaders
from ingestion.text_splitter import create_document_splitter
from ingestion.record_manager import create_record_manager


class IngestionOrchestrator:
    """Main orchestrator for the ingestion pipeline."""
    
    def __init__(self):
        """Initialize the ingestion orchestrator."""
        self.embeddings_model = get_embeddings_model()
        self.vector_store = create_faiss_store(embedding_dim=768)  # Gemini embedding dimension
        self.document_splitter = create_document_splitter()
        self.record_manager = create_record_manager()
        
        logger.info("Initialized ingestion orchestrator")
    
    def run_ingestion(self, force_update: bool = False) -> Dict[str, Any]:
        """Run the complete ingestion pipeline.
        
        Args:
            force_update: Force update even if documents are already indexed
            
        Returns:
            Dictionary with ingestion statistics
        """
        try:
            logger.info("Starting ingestion pipeline")
            
            # Step 1: Load documents from all sources
            all_documents = self._load_all_documents()
            logger.info(f"Loaded {len(all_documents)} documents from all sources")
            
            # Step 2: Filter out already indexed documents (unless force update)
            if not force_update:
                all_documents = self._filter_new_documents(all_documents)
                logger.info(f"After filtering, {len(all_documents)} new documents to process")
            
            if not all_documents:
                logger.info("No new documents to process")
                return {"status": "no_new_documents", "processed": 0}
            
            # Step 3: Split documents into chunks
            chunks = self.document_splitter.split_documents(all_documents)
            logger.info(f"Created {len(chunks)} chunks from {len(all_documents)} documents")
            
            # Step 4: Filter out already indexed chunks
            if not force_update:
                chunks = self._filter_new_chunks(chunks)
                logger.info(f"After filtering, {len(chunks)} new chunks to process")
            
            if not chunks:
                logger.info("No new chunks to process")
                return {"status": "no_new_chunks", "processed": 0}
            
            # Step 5: Create embeddings for chunks
            embeddings = self._create_embeddings(chunks)
            logger.info(f"Created {len(embeddings)} embeddings")
            
            # Step 6: Add to vector store
            self.vector_store.add_documents(chunks, embeddings)
            
            # Step 7: Update record manager
            self._update_record_manager(all_documents, chunks)
            
            # Step 8: Save vector store
            self.vector_store.save_index()
            
            # Get final statistics
            stats = self._get_ingestion_stats()
            
            logger.info("Ingestion pipeline completed successfully")
            return stats
            
        except Exception as e:
            logger.error(f"Error in ingestion pipeline: {e}")
            raise
    
    def _load_all_documents(self) -> List[Dict[str, Any]]:
        """Load documents from all sources."""
        all_documents = []
        loaders = create_document_loaders()
        
        for loader in loaders:
            try:
                documents = loader.load_documents()
                all_documents.extend(documents)
                logger.info(f"Loaded {len(documents)} documents from {loader.__class__.__name__}")
            except Exception as e:
                logger.error(f"Error loading documents from {loader.__class__.__name__}: {e}")
                continue
        
        return all_documents
    
    def _filter_new_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out already indexed documents."""
        new_documents = []
        
        for doc in documents:
            if not self.record_manager.is_document_indexed(doc["metadata"]["source"], doc["content"]):
                new_documents.append(doc)
        
        return new_documents
    
    def _filter_new_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out already indexed chunks."""
        new_chunks = []
        
        for chunk in chunks:
            if not self.record_manager.is_chunk_indexed(chunk["metadata"]["chunk_id"]):
                new_chunks.append(chunk)
        
        return new_chunks
    
    def _create_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """Create embeddings for chunks."""
        texts = [chunk["content"] for chunk in chunks]
        
        # Process in batches to avoid memory issues
        batch_size = 100
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embeddings_model.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def _update_record_manager(self, documents: List[Dict[str, Any]], chunks: List[Dict[str, Any]]):
        """Update record manager with new documents and chunks."""
        # Add documents
        for doc in documents:
            try:
                self.record_manager.add_document(
                    source=doc["metadata"]["source"],
                    content=doc["content"],
                    metadata=doc["metadata"]
                )
            except Exception as e:
                logger.warning(f"Error adding document to record manager: {e}")
        
        # Add chunks
        for chunk in chunks:
            try:
                self.record_manager.add_chunk(
                    document_id=0,  # This would need to be properly linked
                    chunk_id=chunk["metadata"]["chunk_id"],
                    chunk_index=chunk["metadata"]["chunk_index"],
                    content=chunk["content"]
                )
            except Exception as e:
                logger.warning(f"Error adding chunk to record manager: {e}")
    
    def _get_ingestion_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        vector_stats = self.vector_store.get_stats()
        record_stats = self.record_manager.get_stats()
        
        return {
            "status": "success",
            "vector_store": vector_stats,
            "record_manager": record_stats,
            "total_processed": vector_stats["total_vectors"]
        }


def run_ingestion_pipeline(force_update: bool = False) -> Dict[str, Any]:
    """Run the complete ingestion pipeline.
    
    Args:
        force_update: Force update even if documents are already indexed
        
    Returns:
        Dictionary with ingestion statistics
    """
    orchestrator = IngestionOrchestrator()
    return orchestrator.run_ingestion(force_update)