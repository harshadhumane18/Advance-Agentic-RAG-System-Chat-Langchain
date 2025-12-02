# src/agents/agentic_orchestrator.py
"""Main agentic orchestrator that coordinates all components."""

import asyncio
import uuid
from typing import Dict, Any, Optional, AsyncGenerator
from datetime import datetime
from loguru import logger

from config.agentic_settings import agentic_settings
from agents.state import AgentState, InputState, QueryClassification
from agents.query_classifier import query_classifier
from agents.research_planner import research_planner
from agents.researcher import researcher
from agents.response_generator import response_generator
from agents.conversation_memory import memory_manager


class AgenticOrchestrator:
    """Main orchestrator for the agentic RAG system."""
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.query_classifier = query_classifier
        self.research_planner = research_planner
        self.researcher = researcher
        self.response_generator = response_generator
        self.memory_manager = memory_manager
    
    async def process_query(self, input_state: InputState) -> AgentState:
        """Process a user query through the complete agentic pipeline."""
        try:
            # Initialize agent state
            state = await self._initialize_state(input_state)
            
            # Step 1: Query Classification
            state = await self._classify_query(state)
            
            # Step 2: Route based on classification
            if state.query_classification.type == "langchain":
                state = await self._handle_langchain_query(state)
            elif state.query_classification.type == "general":
                state = await self._handle_general_query(state)
            else:  # more_info
                state = await self._handle_more_info_query(state)
            
            # Step 3: Update memory
            await self._update_memory(state)
            
            # Step 4: Finalize response
            state.end_time = datetime.now()
            if state.start_time:
                state.processing_time = (state.end_time - state.start_time).total_seconds()
            
            logger.info(f"Query processed in {state.processing_time:.2f} seconds")
            return state
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return await self._handle_error(state, str(e))
    
    async def process_query_streaming(self, input_state: InputState) -> AsyncGenerator[str, None]:
        """Process a query with streaming response."""
        try:
            # Initialize and classify
            state = await self._initialize_state(input_state)
            state = await self._classify_query(state)
            
            # Handle based on classification
            if state.query_classification.type == "langchain":
                state = await self._handle_langchain_query(state)
            elif state.query_classification.type == "general":
                state = await self._handle_general_query(state)
            else:
                state = await self._handle_more_info_query(state)
            
            # Stream response
            async for chunk in self.response_generator.generate_streaming_response(state):
                yield chunk
            
            # Update memory
            await self._update_memory(state)
            
        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            yield f"Error: {str(e)}"
    
    async def _initialize_state(self, input_state: InputState) -> AgentState:
        """Initialize the agent state."""
        state = AgentState(
            session_id=input_state.session_id,
            current_query=input_state.query,
            streaming_enabled=input_state.streaming,
            start_time=datetime.now()
        )
        
        # Load conversation memory
        session_context = self.memory_manager.get_session_context(input_state.session_id)
        if session_context:
            state.conversation_memory.context = session_context.get("context", {})
            state.conversation_memory.preferences = session_context.get("preferences", {})
        
        # Load conversation history
        history = self.memory_manager.get_conversation_history(
            input_state.session_id, 
            agentic_settings.max_conversation_history
        )
        state.conversation_memory.messages = history
        
        return state
    
    async def _classify_query(self, state: AgentState) -> AgentState:
        """Classify the user query."""
        logger.info("Classifying query...")
        
        classification = await self.query_classifier.classify_query(
            state.current_query,
            state.conversation_memory.context
        )
        
        state.query_classification = classification
        logger.info(f"Query classified as: {classification.type}")
        
        return state
    
    async def _handle_langchain_query(self, state: AgentState) -> AgentState:
        """Handle LangChain-specific queries with research."""
        logger.info("Handling LangChain query with research...")
        
        # Create research plan
        research_plan = await self.research_planner.create_research_plan(
            state.current_query,
            state.conversation_memory.context
        )
        
        state.research_plan = research_plan
        
        # Execute research steps
        all_documents = []
        for i, step in enumerate(research_plan.steps):
            logger.info(f"Executing research step {i+1}/{len(research_plan.steps)}")
            
            # Research the step
            completed_step = await self.researcher.research_step(step)
            all_documents.extend(completed_step.documents)
            
            # Update research plan
            research_plan.steps[i] = completed_step
        
        # Deduplicate and store documents
        state.research_documents = self._deduplicate_documents(all_documents)
        
        # Generate response
        if state.streaming_enabled:
            # For streaming, we'll handle this in the streaming method
            pass
        else:
            state.generated_response = await self.response_generator.generate_response(state)
        
        return state
    
    async def _handle_general_query(self, state: AgentState) -> AgentState:
        """Handle general queries with direct response."""
        logger.info("Handling general query...")
        
        # For general queries, we might still want to do some retrieval
        # but with a simpler approach
        try:
            # Simple retrieval for context
            simple_docs = await self.researcher._retrieve_documents(state.current_query)
            state.research_documents = simple_docs[:5]  # Limit to 5 docs
        except Exception as e:
            logger.warning(f"Error in simple retrieval: {e}")
            state.research_documents = []
        
        # Generate response
        if state.streaming_enabled:
            # For streaming, we'll handle this in the streaming method
            pass
        else:
            state.generated_response = await self.response_generator.generate_response(state)
        
        return state
    
    async def _handle_more_info_query(self, state: AgentState) -> AgentState:
        """Handle queries that need more information."""
        logger.info("Handling more info query...")
        
        # Generate clarification request
        clarification_prompt = f"""The user asked: "{state.current_query}"

Based on the classification reasoning: {state.query_classification.reasoning}

Generate a helpful clarification request that:
1. Acknowledges their question
2. Explains what additional information would be helpful
3. Provides specific examples of what they could ask
4. Maintains a helpful and encouraging tone

Keep it concise and actionable."""

        # Use a simple model call for clarification
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = ChatGoogleGenerativeAI(
            model=agentic_settings.response_model.split("/")[-1],
            google_api_key=agentic_settings.google_api_key,
            temperature=0.3
        )
        
        response = await model.ainvoke([{"role": "user", "content": clarification_prompt}])
        state.generated_response = response.content
        
        return state
    
    async def _update_memory(self, state: AgentState):
        """Update conversation memory."""
        try:
            # Add user message
            from langchain_core.messages import HumanMessage
            user_message = HumanMessage(content=state.current_query)
            self.memory_manager.add_message(state.session_id, user_message)
            
            # Add assistant response
            from langchain_core.messages import AIMessage
            assistant_message = AIMessage(content=state.generated_response)
            self.memory_manager.add_message(
                state.session_id, 
                assistant_message,
                state.response_metadata
            )
            
            # Update context
            context = {
                "last_query_type": state.query_classification.type if state.query_classification else "unknown",
                "research_steps_completed": len(state.research_plan.steps) if state.research_plan else 0,
                "documents_retrieved": len(state.research_documents),
                "processing_time": state.processing_time
            }
            
            self.memory_manager.update_context(state.session_id, context)
            
            # Save research plan if available
            if state.research_plan:
                plan_data = {
                    "plan_id": state.research_plan.plan_id,
                    "query": state.research_plan.query,
                    "steps": [
                        {
                            "step_id": step.step_id,
                            "description": step.description,
                            "status": step.status,
                            "results": step.results
                        }
                        for step in state.research_plan.steps
                    ],
                    "status": state.research_plan.status,
                    "created_at": state.research_plan.created_at.isoformat(),
                    "completed_at": state.research_plan.completed_at.isoformat() if state.research_plan.completed_at else None
                }
                self.memory_manager.save_research_plan(state.session_id, plan_data)
            
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
    
    def _deduplicate_documents(self, documents):
        """Remove duplicate documents."""
        seen_content = set()
        unique_docs = []
        
        for doc in documents:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs
    
    async def _handle_error(self, state: AgentState, error: str) -> AgentState:
        """Handle errors gracefully."""
        state.generated_response = f"I apologize, but I encountered an error while processing your query: {error}. Please try again or rephrase your question."
        state.response_metadata = {"error": error, "response_type": "error"}
        return state


# Global orchestrator instance
agentic_orchestrator = AgenticOrchestrator()