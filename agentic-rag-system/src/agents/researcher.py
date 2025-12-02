# src/agents/researcher.py
"""Enhanced retrieval system with parallel processing and re-ranking."""

import asyncio
from typing import List, Dict, Any, Tuple
from loguru import logger
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.agentic_settings import agentic_settings
from retrieval.faiss_store import create_faiss_store
from embeddings.gemini_embeddings import get_embeddings_model
from agents.state import ResearchStep


class ExpandedQueriesResult(BaseModel):
    """Structured output for expanded queries."""
    queries: List[str] = Field(description="List of 3-5 diverse search queries.")


class RerankedIndicesResult(BaseModel):
    """Structured output for re-ranked document indices."""
    indices: List[int] = Field(description="List of document indices in order of relevance.")


class QueryExpander:
    """Generate multiple search queries from a research step."""
    
    def __init__(self):
        """Initialize the query expander."""
        self.model = ChatGoogleGenerativeAI(
            model=agentic_settings.research_model.split("/")[-1],
            google_api_key=agentic_settings.google_api_key,
            temperature=0.3
        ).with_structured_output(ExpandedQueriesResult)
        self.system_prompt = """You are an expert at generating diverse search queries for document retrieval.

Given a research step, generate 3-5 different search queries that would help find relevant information.
Make the queries:
- Specific and focused
- Use different keywords and phrasings
- Cover different aspects of the topic
- Include technical terms and synonyms

Respond ONLY with a JSON object matching the ExpandedQueriesResult schema."""

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception)
    )
    async def expand_query(self, research_step: str) -> List[str]:
        """Generate multiple search queries from a research step."""
        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"Research step: {research_step}")
            ]
            
            response: ExpandedQueriesResult = await self.model.ainvoke(messages)
            queries = response.queries
            
            logger.info(f"Generated {len(queries)} queries for research step")
            return queries[:5]  # Limit to 5 queries
            
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return [research_step]  # Fallback to original


class DocumentReranker:
    """Re-rank documents by relevance to the query."""
    
    def __init__(self):
        """Initialize the document re-ranker."""
        self.model = ChatGoogleGenerativeAI(
            model=agentic_settings.research_model.split("/")[-1],
            google_api_key=agentic_settings.google_api_key,
            temperature=0.1
        ).with_structured_output(RerankedIndicesResult)
        self.system_prompt = """You are an expert at ranking documents by relevance to a query.

Given a query and a list of documents, rank the documents by their relevance to the query.
Consider:
- Direct relevance to the query topic
- Technical accuracy and depth
- Completeness of information
- Recency and authority of the source

Respond ONLY with a JSON object matching the RerankedIndicesResult schema, containing the document indices in order of relevance (most relevant first)."""

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception)
    )
    async def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """Re-rank documents by relevance to the query."""
        try:
            if len(documents) <= 1:
                return documents
            
            # Prepare document summaries for ranking
            doc_summaries = []
            for i, doc in enumerate(documents):
                summary = f"Doc {i}: {doc.page_content[:200]}... (Source: {doc.metadata.get('source', 'Unknown')})"
                doc_summaries.append(summary)
            
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"Query: {query}\n\nDocuments:\n" + "\n\n".join(doc_summaries))
            ]
            
            response: RerankedIndicesResult = await self.model.ainvoke(messages)
            ranked_indices = response.indices
            
            # Reorder documents based on ranking
            reranked_docs = [documents[i] for i in ranked_indices if i < len(documents)]
            
            logger.info(f"Re-ranked {len(documents)} documents")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error re-ranking documents: {e}")
            return documents  # Return original order on error


class Researcher:
    """Enhanced researcher with parallel retrieval and re-ranking."""
    
    def __init__(self):
        """Initialize the researcher."""
        self.query_expander = QueryExpander()
        self.reranker = DocumentReranker()
        self.vector_store = create_faiss_store(embedding_dim=768)
        self.embeddings_model = get_embeddings_model()
    
    async def research_step(self, research_step: ResearchStep) -> ResearchStep:
        """Research a single step with parallel retrieval and re-ranking."""
        try:
            logger.info(f"Researching step: {research_step.description}")
            
            # Generate multiple queries
            queries = await self.query_expander.expand_query(research_step.description)
            research_step.queries = queries
            
            # Parallel document retrieval
            all_documents = await self._parallel_retrieval(queries)
            
            # Remove duplicates
            unique_documents = self._deduplicate_documents(all_documents)
            
            # Re-rank documents
            reranked_docs = await self.reranker.rerank_documents(
                research_step.description, 
                unique_documents
            )
            
            # Store results
            research_step.documents = reranked_docs[:agentic_settings.max_retrieval_docs]
            research_step.status = "completed"
            
            # Generate step summary
            research_step.results = self._generate_step_summary(research_step)
            
            logger.info(f"Completed research step with {len(research_step.documents)} documents")
            return research_step
            
        except Exception as e:
            logger.error(f"Error researching step: {e}")
            research_step.status = "failed"
            research_step.error = str(e)
            return research_step
    
    async def _parallel_retrieval(self, queries: List[str]) -> List[Document]:
        """Perform parallel document retrieval for multiple queries."""
        try:
            if not agentic_settings.parallel_retrieval:
                # Sequential retrieval
                all_docs = []
                for query in queries:
                    docs = await self._retrieve_documents(query)
                    all_docs.extend(docs)
                return all_docs
            
            # Parallel retrieval
            tasks = [self._retrieve_documents(query) for query in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_docs = []
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Retrieval task failed: {result}")
                else:
                    all_docs.extend(result)
            
            return all_docs
            
        except Exception as e:
            logger.error(f"Error in parallel retrieval: {e}")
            return []
    
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception)
    )
    async def _retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve documents for a single query."""
        try:
            # Create query embedding
            query_embedding = self.embeddings_model.embed_query(query)
            
            # Search vector store
            results = self.vector_store.search(query_embedding, k=agentic_settings.max_retrieval_docs)
            
            # Convert to Document objects
            documents = []
            for result in results:
                doc = Document(
                    page_content=result["metadata"].get("content", ""),
                    metadata=result["metadata"]
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents for query '{query}': {e}")
            return []
    
    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents based on content."""
        seen_content = set()
        unique_docs = []
        
        for doc in documents:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs
    
    def _generate_step_summary(self, research_step: ResearchStep) -> str:
        """Generate a summary of the research step results."""
        if not research_step.documents:
            return "No relevant documents found for this research step."
        
        summary = f"Found {len(research_step.documents)} relevant documents:\n"
        for i, doc in enumerate(research_step.documents[:3], 1):  # Show top 3
            source = doc.metadata.get("source", "Unknown")
            summary += f"{i}. {source}\n"
        
        return summary


# Global researcher instance
researcher = Researcher()