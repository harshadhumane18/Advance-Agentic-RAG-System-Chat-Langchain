# src/config/agentic_settings.py
"""Enhanced configuration for agentic RAG system."""

from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings
from .settings import Settings


class AgenticSettings(Settings):
    """Extended settings for agentic RAG system."""
    
    # LLM Models
    query_model: str = Field("google_genai/gemini-1.5-pro", env="QUERY_MODEL")
    response_model: str = Field("google_genai/gemini-1.5-pro", env="RESPONSE_MODEL")
    research_model: str = Field("google_genai/gemini-1.5-pro", env="RESEARCH_MODEL")
    
    # Agentic Parameters
    max_research_steps: int = Field(5, env="MAX_RESEARCH_STEPS")
    max_retrieval_docs: int = Field(20, env="MAX_RETRIEVAL_DOCS")
    rerank_top_k: int = Field(10, env="RERANK_TOP_K")
    
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
    cache_ttl: int = Field(3600, env="CACHE_TTL")  # 1 hour


# Global agentic settings instance
agentic_settings = AgenticSettings()