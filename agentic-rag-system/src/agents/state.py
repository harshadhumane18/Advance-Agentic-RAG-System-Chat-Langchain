# src/agents/state.py
"""State management for agentic RAG system."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal, Annotated
from datetime import datetime
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig


@dataclass
class QueryClassification:
    """Query classification result."""
    type: Literal["langchain", "general", "more_info"]
    confidence: float
    reasoning: str
    suggested_actions: List[str] = field(default_factory=list)


@dataclass
class ResearchStep:
    """Individual research step."""
    step_id: str
    description: str
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    queries: List[str] = field(default_factory=list)
    documents: List[Document] = field(default_factory=list)
    results: str = ""
    error: Optional[str] = None


@dataclass
class ResearchPlan:
    """Research plan for complex queries."""
    plan_id: str
    query: str
    steps: List[ResearchStep]
    status: Literal["created", "in_progress", "completed", "failed"] = "created"
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class ConversationMemory:
    """Conversation memory and context."""
    session_id: str
    user_id: Optional[str] = None
    messages: List[BaseMessage] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class AgentState:
    """Main agent state."""
    # Core state
    session_id: str
    current_query: str
    query_classification: Optional[QueryClassification] = None
    
    # Research state
    research_plan: Optional[ResearchPlan] = None
    current_step_index: int = 0
    research_documents: List[Document] = field(default_factory=list)
    
    # Memory state
    conversation_memory: ConversationMemory = field(default_factory=lambda: ConversationMemory(session_id=""))
    
    # Response state
    generated_response: str = ""
    response_metadata: Dict[str, Any] = field(default_factory=dict)
    streaming_enabled: bool = True
    
    # Tool state
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance state
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processing_time: Optional[float] = None


@dataclass
class InputState:
    """Input state for the agent."""
    query: str
    session_id: str
    user_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    streaming: bool = True
    config: Optional[RunnableConfig] = None