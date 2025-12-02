"""Configuration settings for the agentic RAG system."""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    
    # Database
    database_url: str = Field("sqlite:///./data/record_manager.db", env="DATABASE_URL")
    
    # FAISS
    faiss_index_path: str = Field("./data/faiss_index", env="FAISS_INDEX_PATH")
    faiss_metadata_path: str = Field("./data/faiss_metadata.pkl", env="FAISS_METADATA_PATH")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: str = Field("./logs/ingestion.log", env="LOG_FILE")
    
    # Data sources
    langchain_sitemap_url: str = Field(
        "https://python.langchain.com/sitemap.xml", 
        env="LANGCHAIN_SITEMAP_URL"
    )
    langgraph_sitemap_url: str = Field(
        "https://langchain-ai.github.io/langgraph/sitemap.xml", 
        env="LANGGRAPH_SITEMAP_URL"
    )
    langsmith_base_url: str = Field(
        "https://docs.smith.langchain.com/", 
        env="LANGSMITH_BASE_URL"
    )
    api_docs_base_url: str = Field(
        "https://api.python.langchain.com/en/latest/", 
        env="API_DOCS_BASE_URL"
    )
    
    # Chunking parameters
    chunk_size: int = Field(4000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    
    # Embedding model
    embedding_model: str = Field("gemini-embedding-001", env="EMBEDDING_MODEL")

    # Agentic Settings (NEW)
    query_model: str = Field("google_genai/gemini-1.5-pro", env="QUERY_MODEL")
    response_model: str = Field("google_genai/gemini-1.5-pro", env="RESPONSE_MODEL")
    research_model: str = Field("google_genai/gemini-1.5-pro", env="RESEARCH_MODEL")
    
    # Memory Settings
    max_conversation_history: int = Field(50, env="MAX_CONVERSATION_HISTORY")
    memory_persistence: bool = Field(True, env="MEMORY_PERSISTENCE")
    
    # Streaming Settings
    enable_streaming: bool = Field(True, env="ENABLE_STREAMING")
    stream_chunk_size: int = Field(100, env="STREAM_CHUNK_SIZE")
    
    # API Settings
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    enable_websocket: bool = Field(True, env="ENABLE_WEBSOCKET")
    
    # Tool Settings
    enable_tools: bool = Field(True, env="ENABLE_TOOLS")
    max_tool_calls: int = Field(3, env="MAX_TOOL_CALLS")
    
    # Performance Settings
    parallel_retrieval: bool = Field(True, env="PARALLEL_RETRIEVAL")
    cache_embeddings: bool = Field(True, env="CACHE_EMBEDDINGS")
    cache_ttl: int = Field(3600, env="CACHE_TTL")
    
    # Research Settings
    max_research_steps: int = Field(5, env="MAX_RESEARCH_STEPS")
    max_retrieval_docs: int = Field(20, env="MAX_RETRIEVAL_DOCS")
    rerank_top_k: int = Field(10, env="RERANK_TOP_K")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()