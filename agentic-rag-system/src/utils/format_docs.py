# src/utils/format_docs.py
"""Document formatting utilities."""

from typing import List
from langchain_core.documents import Document


def format_documents(documents: List[Document], max_length: int = 200) -> str:
    """Format documents for context inclusion."""
    if not documents:
        return "No documents found."
    
    formatted_docs = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        title = doc.metadata.get("title", "Untitled")
        content = doc.page_content[:max_length] + "..." if len(doc.page_content) > max_length else doc.page_content
        
        formatted_doc = f"""
**Document {i}: {title}**
Source: {source}
Content: {content}
"""
        formatted_docs.append(formatted_doc)
    
    return "\n".join(formatted_docs)


def format_research_plan(plan) -> str:
    """Format research plan for display."""
    if not plan or not plan.steps:
        return "No research plan available."
    
    formatted_steps = []
    for i, step in enumerate(plan.steps, 1):
        status_icon = "✅" if step.status == "completed" else "⏳" if step.status == "in_progress" else "⏸️"
        formatted_steps.append(f"{i}. {status_icon} {step.description}")
    
    return "\n".join(formatted_steps)