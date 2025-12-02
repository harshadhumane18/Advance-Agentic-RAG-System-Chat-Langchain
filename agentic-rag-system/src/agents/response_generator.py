# src/agents/response_generator.py
"""Response generation with streaming support."""

import asyncio
from typing import List, Dict, Any, AsyncGenerator, Optional
from loguru import logger
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config.agentic_settings import agentic_settings
from agents.state import AgentState, ResearchPlan
from utils.format_docs import format_documents


class ResponseGenerator:
    """Intelligent response generator with streaming support."""
    
    def __init__(self):
        """Initialize the response generator."""
        self.model = ChatGoogleGenerativeAI(
            model=agentic_settings.response_model.split("/")[-1],
            google_api_key=agentic_settings.google_api_key,
            temperature=0.3
        )
    
    async def generate_response(self, state: AgentState) -> str:
        """Generate a comprehensive response based on the agent state."""
        try:
            logger.info("Generating response...")
            
            # Prepare context
            context = self._prepare_context(state)
            
            # Create system prompt
            system_prompt = self._create_system_prompt(state)
            
            # Generate response
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=state.current_query)
            ]
            
            response = await self.model.ainvoke(messages)
            
            # Update state
            state.generated_response = response.content
            state.response_metadata = {
                "response_type": "comprehensive",
                "sources_used": len(state.research_documents),
                "research_steps": len(state.research_plan.steps) if state.research_plan else 0
            }
            
            logger.info("Response generated successfully")
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_fallback_response(state)
    
    async def generate_streaming_response(self, state: AgentState) -> AsyncGenerator[str, None]:
        """Generate a streaming response."""
        try:
            logger.info("Generating streaming response...")
            context = self._prepare_context(state)
            system_prompt = self._create_system_prompt(state)

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=state.current_query)
            ]

            full_response = ""
            async for chunk in self.model.astream(messages):
                if chunk.content:
                    full_response += chunk.content
                    yield chunk.content
                    await asyncio.sleep(0.01)

            state.generated_response = full_response
            state.response_metadata = {
                "response_type": "streaming",
                "sources_used": len(state.research_documents),
                "research_steps": len(state.research_plan.steps) if state.research_plan else 0
            }
            logger.info("Streaming response completed")
            
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            fallback = self._generate_fallback_response(state)
            for char in fallback:
                yield char
                await asyncio.sleep(0.01)
    
    def _prepare_context(self, state: AgentState) -> str:
        """Prepare context from research documents and conversation history."""
        context_parts = []
        
        # Add research documents
        if state.research_documents:
            context_parts.append("## Research Documents")
            context_parts.append(format_documents(state.research_documents[:10]))  # Top 10 docs
        
        # Add research plan if available
        if state.research_plan:
            context_parts.append("## Research Plan")
            for i, step in enumerate(state.research_plan.steps, 1):
                status_icon = "✅" if step.status == "completed" else "⏳" if step.status == "in_progress" else "⏸️"
                context_parts.append(f"{i}. {status_icon} {step.description}")
        
        # Add conversation context
        if state.conversation_memory.messages:
            context_parts.append("## Previous Conversation")
            recent_messages = state.conversation_memory.messages[-4:]  # Last 4 messages
            for msg in recent_messages:
                role = "User" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
                context_parts.append(f"{role}: {msg.content[:200]}...")
        
        return "\n\n".join(context_parts)
    
    def _create_system_prompt(self, state: AgentState) -> str:
        """Create system prompt based on query type and context."""
        base_prompt = """You are an expert LangChain documentation assistant. You provide accurate, helpful, and comprehensive answers about LangChain, LangGraph, LangSmith, and related technologies.

Guidelines:
- Provide accurate, up-to-date information
- Include code examples when relevant
- Reference specific documentation sources
- Be concise but comprehensive
- Use markdown formatting for code and structure
- If you're unsure about something, say so clearly

Context Information:
{context}

Answer the user's question based on the provided context and your knowledge."""

        # Customize based on query type
        if state.query_classification and state.query_classification.type == "langchain":
            base_prompt += "\n\nThis is a LangChain-specific question. Provide detailed technical information with code examples and best practices."
        elif state.query_classification and state.query_classification.type == "general":
            base_prompt += "\n\nThis is a general question. Provide a helpful answer while noting that you specialize in LangChain technologies."
        else:
            base_prompt += "\n\nThis question needs clarification. Ask for more specific information to provide a better answer."

        return base_prompt.format(context=self._prepare_context(state))
    
    def _generate_fallback_response(self, state: AgentState) -> str:
        """Generate a fallback response when main generation fails."""
        if state.query_classification and state.query_classification.type == "more_info":
            return "I'd be happy to help! Could you please provide more specific details about what you're looking for? For example, are you asking about a specific LangChain component, implementation, or concept?"
        elif state.research_documents:
            return f"I found some relevant information in our documentation. Here's what I can tell you based on the {len(state.research_documents)} documents I found: [Response generation failed, but documents were retrieved]"
        else:
            return "I apologize, but I'm having trouble generating a response right now. Please try rephrasing your question or ask for help with a specific LangChain topic."


# Global response generator instance
response_generator = ResponseGenerator()