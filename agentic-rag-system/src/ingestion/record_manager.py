"""Record manager for tracking indexed documents."""

import sqlite3
from typing import List, Dict, Any, Set, Optional
from pathlib import Path
from loguru import logger
import hashlib
import json

from config.settings import settings


class SQLRecordManager:
    """SQLite-based record manager for tracking indexed documents."""
    
    def __init__(self, db_path: str = None):
        """Initialize record manager.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path or settings.database_url.replace("sqlite:///", "")
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        logger.info(f"Initialized SQLRecordManager with database: {self.db_path}")
    
    def _init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    metadata TEXT,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source, content_hash)
                )
            """)
            
            # Create chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER,
                    chunk_id TEXT NOT NULL,
                    chunk_index INTEGER,
                    content_hash TEXT NOT NULL,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id),
                    UNIQUE(chunk_id)
                )
            """)
            
            conn.commit()
    
    def add_document(self, source: str, content: str, metadata: Dict[str, Any]) -> int:
        """Add a document to the record manager.
        
        Args:
            source: Document source URL
            content: Document content
            metadata: Document metadata
            
        Returns:
            Document ID
        """
        content_hash = self._hash_content(content)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT INTO documents (source, content_hash, metadata)
                    VALUES (?, ?, ?)
                """, (source, content_hash, json.dumps(metadata)))
                
                document_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Added document {source} with ID {document_id}")
                return document_id
                
            except sqlite3.IntegrityError:
                # Document already exists
                cursor.execute("""
                    SELECT id FROM documents 
                    WHERE source = ? AND content_hash = ?
                """, (source, content_hash))
                
                result = cursor.fetchone()
                if result:
                    logger.info(f"Document {source} already exists with ID {result[0]}")
                    return result[0]
                else:
                    raise
    
    def add_chunk(self, document_id: int, chunk_id: str, chunk_index: int, content: str) -> int:
        """Add a chunk to the record manager.
        
        Args:
            document_id: Parent document ID
            chunk_id: Unique chunk identifier
            chunk_index: Index of chunk in document
            content: Chunk content
            
        Returns:
            Chunk ID
        """
        content_hash = self._hash_content(content)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT INTO chunks (document_id, chunk_id, chunk_index, content_hash)
                    VALUES (?, ?, ?, ?)
                """, (document_id, chunk_id, chunk_index, content_hash))
                
                chunk_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Added chunk {chunk_id} for document {document_id}")
                return chunk_id
                
            except sqlite3.IntegrityError:
                # Chunk already exists
                cursor.execute("""
                    SELECT id FROM chunks WHERE chunk_id = ?
                """, (chunk_id,))
                
                result = cursor.fetchone()
                if result:
                    logger.info(f"Chunk {chunk_id} already exists with ID {result[0]}")
                    return result[0]
                else:
                    raise
    
    def is_document_indexed(self, source: str, content: str) -> bool:
        """Check if a document is already indexed.
        
        Args:
            source: Document source URL
            content: Document content
            
        Returns:
            True if document is indexed
        """
        content_hash = self._hash_content(content)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM documents 
                WHERE source = ? AND content_hash = ?
            """, (source, content_hash))
            
            count = cursor.fetchone()[0]
            return count > 0
    
    def is_chunk_indexed(self, chunk_id: str) -> bool:
        """Check if a chunk is already indexed.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            True if chunk is indexed
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM chunks WHERE chunk_id = ?
            """, (chunk_id,))
            
            count = cursor.fetchone()[0]
            return count > 0
    
    def get_indexed_sources(self) -> Set[str]:
        """Get all indexed source URLs.
        
        Returns:
            Set of indexed source URLs
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT source FROM documents")
            
            sources = {row[0] for row in cursor.fetchall()}
            return sources
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed documents.
        
        Returns:
            Dictionary with statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count documents
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            
            # Count chunks
            cursor.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cursor.fetchone()[0]
            
            # Count unique sources
            cursor.execute("SELECT COUNT(DISTINCT source) FROM documents")
            source_count = cursor.fetchone()[0]
            
            return {
                "total_documents": doc_count,
                "total_chunks": chunk_count,
                "unique_sources": source_count
            }
    
    def _hash_content(self, content: str) -> str:
        """Generate hash for content.
        
        Args:
            content: Content to hash
            
        Returns:
            Content hash
        """
        return hashlib.md5(content.encode('utf-8')).hexdigest()


def create_record_manager() -> SQLRecordManager:
    """Create a record manager instance.
    
    Returns:
        SQLRecordManager instance
    """
    return SQLRecordManager()