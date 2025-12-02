"""Constants for the agentic RAG system."""

# FAISS Index Names
FAISS_INDEX_NAME = "agentic_rag_documents"
FAISS_METADATA_INDEX_NAME = "agentic_rag_metadata"

# Document types
DOCUMENT_TYPES = {
    "langchain_docs": "langchain_documentation",
    "langgraph_docs": "langgraph_documentation", 
    "langsmith_docs": "langsmith_documentation",
    "api_docs": "api_documentation"
}

# Supported file types
SUPPORTED_EXTENSIONS = {".html", ".htm", ".md", ".txt", ".pdf", ".docx"}

# Metadata keys
METADATA_KEYS = ["source", "title", "description", "language", "document_type", "chunk_id"]