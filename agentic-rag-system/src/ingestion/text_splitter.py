"""Text splitting utilities for document chunking."""

from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config.settings import settings


class DocumentSplitter:
    """Document splitter for creating text chunks."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """Initialize document splitter.
        
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info(f"Initialized text splitter: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into chunks.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of chunk dictionaries
        """
        try:
            logger.info(f"Splitting {len(documents)} documents into chunks")
            
            all_chunks = []
            for doc in documents:
                chunks = self._split_single_document(doc)
                all_chunks.extend(chunks)
            
            # Filter out very short chunks
            filtered_chunks = [chunk for chunk in all_chunks if len(chunk["content"]) > 10]
            
            logger.info(f"Created {len(filtered_chunks)} chunks from {len(documents)} documents")
            return filtered_chunks
            
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise
    
    def _split_single_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a single document into chunks.
        
        Args:
            document: Document dictionary
            
        Returns:
            List of chunk dictionaries
        """
        content = document["content"]
        metadata = document["metadata"].copy()
        
        # Split text into chunks
        text_chunks = self.splitter.split_text(content)
        
        # Create chunk documents
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = f"{metadata['source']}_chunk_{i}"
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(text_chunks)
            
            chunks.append({
                "content": chunk_text,
                "metadata": chunk_metadata
            })
        
        return chunks


def create_document_splitter() -> DocumentSplitter:
    """Create a document splitter instance.
    
    Returns:
        DocumentSplitter instance
    """
    return DocumentSplitter()