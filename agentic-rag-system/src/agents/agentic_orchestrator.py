# src/agents/agentic_orchestrator.py
"""LangGraph-based orchestrator for Agentic RAG."""

import asyncio
from datetime import datetime
from loguru import logger
from typing import Dict, Any

from config.agentic_settings import agentic_settings
from agents.state import AgentState, InputState
from agents.query_classifier import query_classifier
from agents.research_planner import research_planner
from agents.researcher import researcher
from agents.response_generator import response_generator
from agents.conversation_memory import memory_manager

# LangGraph imports - Using Memory Checkpointer for compatibility
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# -----------------------------
# Small resilience helpers
# -----------------------------
async def with_timeout(coro, seconds: float = 30.0):
    return await asyncio.wait_for(coro, timeout=seconds)


# -----------------------------
# Node definitions
# -----------------------------
async def initialize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize the agent state with memory."""
    input_state: InputState = state["input_state"]

    agent_state = AgentState(
        session_id=input_state.session_id,
        current_query=input_state.query,
        streaming_enabled=input_state.streaming,
        start_time=datetime.now()
    )

    # Load memory
    session_context = memory_manager.get_session_context(input_state.session_id)
    if session_context:
        agent_state.conversation_memory.context = session_context.get("context", {})
        agent_state.conversation_memory.preferences = session_context.get("preferences", {})

    history = memory_manager.get_conversation_history(
        input_state.session_id,
        agentic_settings.max_conversation_history
    )
    agent_state.conversation_memory.messages = history

    state["agent_state"] = agent_state
    return state


async def classify_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """Classify the query type."""
    agent_state: AgentState = state["agent_state"]
    logger.info("Classifying query...")

    classification = await query_classifier.classify_query(
        agent_state.current_query,
        agent_state.conversation_memory.context
    )
    agent_state.query_classification = classification
    state["agent_state"] = agent_state
    return state


async def handle_langchain_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle LangChain-specific queries with research."""
    agent_state: AgentState = state["agent_state"]
    logger.info("Handling LangChain query...")

    # Create research plan (with timeout)
    research_plan = await with_timeout(
        research_planner.create_research_plan(
            agent_state.current_query,
            agent_state.conversation_memory.context
        ),
        seconds=30.0
    )
    agent_state.research_plan = research_plan

    # Parallel research over steps
    tasks = [researcher.research_step(step) for step in research_plan.steps]
    completed_steps = await asyncio.gather(*tasks, return_exceptions=True)

    all_documents = []
    for i, result in enumerate(completed_steps):
        if isinstance(result, Exception):
            logger.warning(f"Research step {i+1} failed: {result}")
            research_plan.steps[i].status = "failed"
            research_plan.steps[i].error = str(result)
        else:
            research_plan.steps[i] = result
            all_documents.extend(result.documents)

    agent_state.research_documents = deduplicate_documents(all_documents)

    # Generate response only if not streaming; streaming happens in streaming node
    if not agent_state.streaming_enabled:
        txt = await response_generator.generate_response(agent_state)
        agent_state.generated_response = txt
        agent_state.response_metadata = {
            "response_type": "comprehensive",
            "sources_used": len(agent_state.research_documents),
            "research_steps": len(agent_state.research_plan.steps) if agent_state.research_plan else 0
        }

    state["agent_state"] = agent_state
    return state


async def handle_general_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle general queries with direct retrieval."""
    agent_state: AgentState = state["agent_state"]
    logger.info("Handling general query...")

    try:
        docs = await researcher._retrieve_documents(agent_state.current_query)
        agent_state.research_documents = docs[:5]
    except Exception as e:
        logger.warning(f"Simple retrieval failed: {e}")
        agent_state.research_documents = []

    # Generate response only if not streaming
    if not agent_state.streaming_enabled:
        txt = await response_generator.generate_response(agent_state)
        agent_state.generated_response = txt
        agent_state.response_metadata = {
            "response_type": "comprehensive",
            "sources_used": len(agent_state.research_documents),
            "research_steps": len(agent_state.research_plan.steps) if agent_state.research_plan else 0
        }

    state["agent_state"] = agent_state
    return state


async def handle_more_info_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle queries needing clarification."""
    agent_state: AgentState = state["agent_state"]
    logger.info("Handling more info query...")

    clarification_prompt = f"""The user asked: "{agent_state.current_query}"

    Based on the classification reasoning: {agent_state.query_classification.reasoning}

    Generate a helpful clarification request that:
    1. Acknowledges their question
    2. Explains what additional information would be helpful
    3. Provides specific examples of what they could ask
    4. Maintains a helpful and encouraging tone

    Keep it concise and actionable."""
    response = await response_generator.model.ainvoke([{"role": "user", "content": clarification_prompt}])
    agent_state.generated_response = response.content
    agent_state.response_metadata = {"response_type": "clarification"}

    state["agent_state"] = agent_state
    return state


async def streaming_generate(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a streaming response and store the full text."""
    agent_state: AgentState = state["agent_state"]
    logger.info("Streaming generation...")

    full = ""
    async for chunk in response_generator.generate_streaming_response(agent_state):
        full += chunk

    agent_state.generated_response = full
    agent_state.response_metadata = {
        "response_type": "streaming",
        "sources_used": len(agent_state.research_documents),
        "research_steps": len(agent_state.research_plan.steps) if agent_state.research_plan else 0
    }

    state["agent_state"] = agent_state
    return state


async def handle_generate(state: Dict[str, Any]) -> Dict[str, Any]:
    """Non-streaming finalize (no-op if already generated)."""
    agent_state: AgentState = state["agent_state"]

    # If something hasn't generated yet (edge case), generate now
    if not agent_state.generated_response:
        txt = await response_generator.generate_response(agent_state)
        agent_state.generated_response = txt

    if not agent_state.response_metadata:
        agent_state.response_metadata = {
            "response_type": "comprehensive",
            "sources_used": len(agent_state.research_documents),
            "research_steps": len(agent_state.research_plan.steps) if agent_state.research_plan else 0
        }

    state["agent_state"] = agent_state
    return state


async def finalize(state: Dict[str, Any]) -> Dict[str, Any]:
    """Set end_time and processing_time."""
    agent_state: AgentState = state["agent_state"]
    agent_state.end_time = datetime.now()
    if agent_state.start_time:
        agent_state.processing_time = (agent_state.end_time - agent_state.start_time).total_seconds()
    state["agent_state"] = agent_state
    return state


async def update_memory(state: Dict[str, Any]) -> Dict[str, Any]:
    """Save conversation history, context, and research plan."""
    agent_state: AgentState = state["agent_state"]

    try:
        from langchain_core.messages import HumanMessage, AIMessage
        memory_manager.add_message(agent_state.session_id, HumanMessage(content=agent_state.current_query))
        memory_manager.add_message(
            agent_state.session_id,
            AIMessage(content=agent_state.generated_response),
            agent_state.response_metadata or {}
        )

        context = {
            "last_query_type": agent_state.query_classification.type if agent_state.query_classification else "unknown",
            "research_steps_completed": len(agent_state.research_plan.steps) if agent_state.research_plan else 0,
            "documents_retrieved": len(agent_state.research_documents),
            "processing_time": agent_state.processing_time,
        }
        memory_manager.update_context(agent_state.session_id, context)

        if agent_state.research_plan:
            plan = agent_state.research_plan
            plan_data = {
                "plan_id": plan.plan_id,
                "query": plan.query,
                "steps": [
                    {
                        "step_id": s.step_id,
                        "description": s.description,
                        "status": s.status,
                        "results": s.results,
                    }
                    for s in plan.steps
                ],
                "status": plan.status,
                "created_at": plan.created_at.isoformat(),
                "completed_at": plan.completed_at.isoformat() if plan.completed_at else None,
            }
            memory_manager.save_research_plan(agent_state.session_id, plan_data)

    except Exception as e:
        logger.error(f"Error updating memory: {e}")

    return state


def deduplicate_documents(documents):
    """Remove duplicate documents."""
    seen_content = set()
    unique_docs = []
    for doc in documents:
        content_hash = hash(doc.page_content)
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_docs.append(doc)
    return unique_docs


# -----------------------------
# Build LangGraph
# -----------------------------
graph = StateGraph(dict)

graph.add_node("initialize_state", initialize_state)
graph.add_node("classify_query", classify_query)
graph.add_node("handle_langchain", handle_langchain_query)
graph.add_node("handle_general", handle_general_query)
graph.add_node("handle_more_info", handle_more_info_query)
graph.add_node("streaming_generate", streaming_generate)
graph.add_node("handle_generate", handle_generate)
graph.add_node("finalize", finalize)
graph.add_node("update_memory", update_memory)

graph.set_entry_point("initialize_state")
graph.add_edge("initialize_state", "classify_query")

# Conditional routing after classification
def route_classification(state: Dict[str, Any]) -> str:
    classification = state["agent_state"].query_classification.type
    if classification == "langchain":
        return "handle_langchain"
    elif classification == "general":
        return "handle_general"
    else:
        return "handle_more_info"

graph.add_conditional_edges(
    "classify_query",
    route_classification,
    {
        "handle_langchain": "handle_langchain",
        "handle_general": "handle_general",
        "handle_more_info": "handle_more_info"
    }
)

# Conditional generation route: streaming vs non-streaming
def route_generation(state: Dict[str, Any]) -> str:
    streaming = state["agent_state"].streaming_enabled
    return "streaming_generate" if streaming else "handle_generate"

graph.add_conditional_edges(
    "handle_langchain",
    route_generation,
    {"streaming_generate": "streaming_generate", "handle_generate": "handle_generate"}
)
graph.add_conditional_edges(
    "handle_general",
    route_generation,
    {"streaming_generate": "streaming_generate", "handle_generate": "handle_generate"}
)
graph.add_conditional_edges(
    "handle_more_info",
    route_generation,
    {"streaming_generate": "streaming_generate", "handle_generate": "handle_generate"}
)

# finalize -> memory -> END
graph.add_edge("handle_generate", "finalize")
graph.add_edge("streaming_generate", "finalize")
graph.add_edge("finalize", "update_memory")
graph.add_edge("update_memory", END)


# Global checkpointer instance for persistence across calls
_memory_checkpointer = None

def get_memory_checkpointer():
    """Get or create the global memory checkpointer."""
    global _memory_checkpointer
    if _memory_checkpointer is None:
        _memory_checkpointer = MemorySaver()
    return _memory_checkpointer


# Thin wrapper to use from main.py
async def run_with_graph(input_state: InputState) -> AgentState:
    """Run the agentic graph with proper checkpointing."""

    # Use Memory Checkpointer - works with all LangGraph versions
    checkpointer = get_memory_checkpointer()
    agentic_app = graph.compile(checkpointer=checkpointer)
    
    # Industry standard: Use thread_id for conversation persistence
    config = {
        "configurable": {
            "thread_id": input_state.session_id
        }
    }
    
    try:
        out = await agentic_app.ainvoke(
            {"input_state": input_state}, 
            config=config
        )
        return out["agent_state"]
    except Exception as e:
        logger.error(f"Error in agentic graph: {e}")
        # Return a graceful fallback
        return AgentState(
            session_id=input_state.session_id,
            current_query=input_state.query,
            generated_response="I apologize, but I'm having trouble processing your request right now. Please try again.",
            response_metadata={"error": str(e)}
        )