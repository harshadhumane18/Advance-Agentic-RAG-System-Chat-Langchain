# src/api/server.py
"""FastAPI server for Agentic RAG System."""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from loguru import logger
import uvicorn

# Add src to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.agentic_orchestrator import run_with_graph
from agents.state import InputState, AgentState
from config.agentic_settings import agentic_settings


# -----------------------------
# Pydantic Models for API
# -----------------------------

class ChatMessage(BaseModel):
    """Chat message model."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str
    session_id: Optional[str] = None
    stream: bool = True
    model: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response model."""
    message: str
    session_id: str
    timestamp: datetime
    metadata: Dict[str, Any]


class ThreadCreateRequest(BaseModel):
    """Thread creation request."""
    user_id: str
    title: Optional[str] = None


class ThreadResponse(BaseModel):
    """Thread response model."""
    thread_id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class AgentStateUpdate(BaseModel):
    """Agent state update for streaming."""
    step: str
    progress: int
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime


# -----------------------------
# State Management
# -----------------------------

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, List[str]] = {}
    
    async def connect(self, websocket: WebSocket, connection_id: str, session_id: str):
        """Connect a WebSocket."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        if session_id not in self.session_connections:
            self.session_connections[session_id] = []
        self.session_connections[session_id].append(connection_id)
        logger.info(f"WebSocket connected: {connection_id} for session {session_id}")
    
    def disconnect(self, connection_id: str, session_id: str):
        """Disconnect a WebSocket."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        if session_id in self.session_connections:
            self.session_connections[session_id].remove(connection_id)
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_to_session(self, session_id: str, data: Dict[str, Any]):
        """Send data to all connections in a session."""
        if session_id in self.session_connections:
            for connection_id in self.session_connections[session_id]:
                if connection_id in self.active_connections:
                    try:
                        await self.active_connections[connection_id].send_text(json.dumps(data))
                    except Exception as e:
                        logger.error(f"Error sending to {connection_id}: {e}")
                        self.disconnect(connection_id, session_id)


# Global connection manager
manager = ConnectionManager()

# -----------------------------
# State Adapters
# -----------------------------

def agent_state_to_langchain_messages(agent_state: AgentState) -> List[Dict[str, Any]]:
    """Convert AgentState to LangChain message format."""
    messages = []
    
    # Add conversation history
    if agent_state.conversation_memory.messages:
        for msg in agent_state.conversation_memory.messages:
            messages.append({
                "role": "user" if msg.role == "user" else "assistant",
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None
            })
    
    # Add current query
    messages.append({
        "role": "user",
        "content": agent_state.current_query,
        "timestamp": datetime.now().isoformat()
    })
    
    return messages


def create_agent_state_update(step: str, progress: int, message: str, 
                            data: Optional[Dict[str, Any]] = None) -> AgentStateUpdate:
    """Create an agent state update."""
    return AgentStateUpdate(
        step=step,
        progress=progress,
        message=message,
        data=data,
        timestamp=datetime.now()
    )


# -----------------------------
# FastAPI App
# -----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan manager."""
    logger.info("Starting Agentic RAG API Server...")
    yield
    logger.info("Shutting down Agentic RAG API Server...")


app = FastAPI(
    title="Agentic RAG API",
    description="API for Agentic RAG System with LangGraph",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# API Endpoints
# -----------------------------

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/threads", response_model=ThreadResponse)
async def create_thread(request: ThreadCreateRequest):
    """Create a new conversation thread."""
    thread_id = f"thread_{uuid.uuid4().hex[:8]}"
    now = datetime.now()
    
    thread = ThreadResponse(
        thread_id=thread_id,
        user_id=request.user_id,
        title=request.title or f"Chat {now.strftime('%Y-%m-%d %H:%M')}",
        created_at=now,
        updated_at=now
    )
    
    logger.info(f"Created thread {thread_id} for user {request.user_id}")
    return thread


@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str):
    """Get thread information."""
    # In a real implementation, you'd fetch from database
    return {"thread_id": thread_id, "status": "active"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message (non-streaming)."""
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
    
    try:
        # Create input state
        input_state = InputState(
            query=request.message,
            session_id=session_id,
            streaming=False
        )
        
        # Process with agentic orchestrator
        result = await run_with_graph(input_state)
        
        # Convert to response format
        response = ChatResponse(
            message=result.generated_response,
            session_id=session_id,
            timestamp=datetime.now(),
            metadata={
                "query_classification": result.query_classification.type if result.query_classification else "unknown",
                "research_steps": len(result.research_plan.steps) if result.research_plan else 0,
                "documents_used": len(result.research_documents),
                "processing_time": result.processing_time
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Process a chat message with streaming response."""
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
    
    async def generate_stream():
        """Generate streaming response."""
        try:
            # Send initial state
            yield f"data: {json.dumps(create_agent_state_update('initializing', 0, 'Starting agentic processing...').dict())}\n\n"
            
            # Create input state
            input_state = InputState(
                query=request.message,
                session_id=session_id,
                streaming=True
            )
            
            # Process with streaming
            async for update in process_with_streaming(input_state):
                yield f"data: {json.dumps(update.dict())}\n\n"
            
            # Send completion
            yield f"data: {json.dumps(create_agent_state_update('completed', 100, 'Processing complete').dict())}\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield f"data: {json.dumps(create_agent_state_update('error', 0, f'Error: {str(e)}').dict())}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


async def process_with_streaming(input_state: InputState) -> AsyncGenerator[AgentStateUpdate, None]:
    """Process input with streaming updates."""
    try:
        # Initialize state
        yield create_agent_state_update("initialize", 10, "Initializing agent state...")
        
        # Classify query
        yield create_agent_state_update("classify", 20, "Classifying query type...")
        
        # Process with orchestrator (this would need to be modified to support streaming)
        result = await run_with_graph(input_state)
        
        # Send research plan updates
        if result.research_plan:
            yield create_agent_state_update(
                "research_plan", 
                30, 
                f"Created research plan with {len(result.research_plan.steps)} steps",
                {"steps": [{"description": step.description, "status": step.status} for step in result.research_plan.steps]}
            )
        
        # Send research progress
        yield create_agent_state_update(
            "research", 
            60, 
            f"Retrieved {len(result.research_documents)} documents",
            {"document_count": len(result.research_documents)}
        )
        
        # Send response generation
        yield create_agent_state_update(
            "generate", 
            80, 
            "Generating response...",
            {"response_length": len(result.generated_response)}
        )
        
        # Send final response
        yield create_agent_state_update(
            "complete", 
            100, 
            "Response generated successfully",
            {
                "response": result.generated_response,
                "metadata": {
                    "query_classification": result.query_classification.type if result.query_classification else "unknown",
                    "research_steps": len(result.research_plan.steps) if result.research_plan else 0,
                    "documents_used": len(result.research_documents),
                    "processing_time": result.processing_time
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error in streaming processing: {e}")
        yield create_agent_state_update("error", 0, f"Processing error: {str(e)}")


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time updates."""
    connection_id = f"conn_{uuid.uuid4().hex[:8]}"
    await manager.connect(websocket, connection_id, session_id)
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "chat":
                # Process chat message
                await process_chat_message(session_id, message.get("content", ""))
            elif message.get("type") == "ping":
                # Respond to ping
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}))
                
    except WebSocketDisconnect:
        manager.disconnect(connection_id, session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(connection_id, session_id)


async def process_chat_message(session_id: str, content: str):
    """Process a chat message and send updates via WebSocket."""
    try:
        # Send initial update
        await manager.send_to_session(session_id, create_agent_state_update(
            "start", 0, "Processing your message..."
        ).dict())
        
        # Create input state
        input_state = InputState(
            query=content,
            session_id=session_id,
            streaming=True
        )
        
        # Process with streaming updates
        async for update in process_with_streaming(input_state):
            await manager.send_to_session(session_id, update.dict())
            
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        await manager.send_to_session(session_id, create_agent_state_update(
            "error", 0, f"Error: {str(e)}"
        ).dict())


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=agentic_settings.api_host,
        port=agentic_settings.api_port,
        reload=True,
        log_level="info"
    )