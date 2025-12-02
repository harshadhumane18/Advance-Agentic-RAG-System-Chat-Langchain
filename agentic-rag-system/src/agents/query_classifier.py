# src/agents/query_classifier.py
"""Query classification and routing system."""

import asyncio
from typing import Dict, Any, List
from loguru import logger
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.agentic_settings import agentic_settings
from agents.state import QueryClassification


class QueryClassificationResult(BaseModel):
    """Structured output for query classification."""
    type: str
    confidence: float
    reasoning: str
    suggested_actions: List[str]


class QueryClassifier:
    """Intelligent query classifier and router."""
    
    def __init__(self):
        """Initialize the query classifier."""
        self.model = ChatGoogleGenerativeAI(
            model=agentic_settings.query_model.split("/")[-1],
            google_api_key=agentic_settings.google_api_key,
            temperature=0.1
        ).with_structured_output(QueryClassificationResult)
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        return """You are an expert query classifier for a LangChain documentation assistant. Your task is to classify user queries and respond ONLY with a JSON object matching the QueryClassificationResult schema.

        Your task is to classify user queries into one of three categories:

        1. **langchain**: Questions specifically about LangChain, LangGraph, LangSmith, or related technologies
        - Examples: "How do I use LangChain chains?", "What is LangGraph?", "How to implement RAG with LangChain?"
        - These require research and detailed technical responses

        2. **general**: General questions not related to LangChain
        - Examples: "What is the weather?", "Tell me a joke", "What is machine learning?"
        - These can be answered directly without research

        3. **more_info**: Queries that are unclear, too vague, or need clarification
        - Examples: "Help me", "I have a problem", "How do I do this?"
        - These require asking the user for more specific information

        Output a JSON object with 'type', 'confidence', 'reasoning', and 'suggested_actions' fields."""

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception) # More specific exceptions are better
    )
    
    async def classify_query(self, query: str, context: Dict[str, Any] = None) -> QueryClassification:
        """Classify a user query."""
        try:
            logger.info(f"Classifying query: {query[:100]}...")
            
            # Prepare messages
            messages = [
                ("system", self.system_prompt),
                HumanMessage(content=f"Query: {query}\nContext: {context or {}}")
            ]
            
            # Get structured output
            classification_result: QueryClassificationResult = await self.model.ainvoke(messages)
            
            result = QueryClassification(
                type=classification_result.type,
                confidence=classification_result.confidence,
                reasoning=classification_result.reasoning,
                suggested_actions=classification_result.suggested_actions
            )
            
            logger.info(f"Query classified as: {result.type} (confidence: {result.confidence})")
            return result
            
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            # Fallback to general classification
            return QueryClassification(
                type="general",
                confidence=0.5,
                reasoning="Error in classification, defaulting to general",
                suggested_actions=["Provide a general response"]
            )


# Global classifier instance
query_classifier = QueryClassifier()