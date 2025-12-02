# src/memory/conversation_memory.py
"""Conversation memory and context management."""

import json
import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from config.agentic_settings import agentic_settings


class ConversationMemoryManager:
    """Manages conversation memory and context persistence."""
    
    def __init__(self, db_path: str = "./data/conversation_memory.db"):
        """Initialize the memory manager."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize the conversation memory database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    created_at TIMESTAMP,
                    last_updated TIMESTAMP,
                    context TEXT,
                    preferences TEXT
                )
            """)
            
            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    message_type TEXT,
                    content TEXT,
                    metadata TEXT,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES conversations (session_id)
                )
            """)
            
            # Research plans table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS research_plans (
                    plan_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    query TEXT,
                    steps TEXT,
                    status TEXT,
                    created_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES conversations (session_id)
                )
            """)
            
            conn.commit()
    
    def create_session(self, session_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new conversation session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO conversations 
                    (session_id, user_id, created_at, last_updated, context, preferences)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    user_id,
                    datetime.now(),
                    datetime.now(),
                    json.dumps({}),
                    json.dumps({})
                ))
                
                conn.commit()
                
            logger.info(f"Created session: {session_id}")
            return {
                "session_id": session_id,
                "user_id": user_id,
                "created_at": datetime.now(),
                "context": {},
                "preferences": {}
            }
            
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return {}
    
    def add_message(self, session_id: str, message: BaseMessage, metadata: Dict[str, Any] = None):
        """Add a message to the conversation."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                message_type = message.__class__.__name__
                content = message.content
                metadata_json = json.dumps(metadata or {})
                
                cursor.execute("""
                    INSERT INTO messages 
                    (session_id, message_type, content, metadata, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (session_id, message_type, content, metadata_json, datetime.now()))
                
                # Update last_updated
                cursor.execute("""
                    UPDATE conversations 
                    SET last_updated = ? 
                    WHERE session_id = ?
                """, (datetime.now(), session_id))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error adding message: {e}")
    
    def get_conversation_history(self, session_id: str, limit: int = None) -> List[BaseMessage]:
        """Get conversation history for a session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT message_type, content, metadata, timestamp
                    FROM messages 
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor.execute(query, (session_id,))
                rows = cursor.fetchall()
                
                messages = []
                for row in rows:
                    message_type, content, metadata_json, timestamp = row
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    
                    if message_type == "HumanMessage":
                        messages.append(HumanMessage(content=content))
                    elif message_type == "AIMessage":
                        messages.append(AIMessage(content=content))
                
                return messages
                
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    def update_context(self, session_id: str, context: Dict[str, Any]):
        """Update session context."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE conversations 
                    SET context = ?, last_updated = ?
                    WHERE session_id = ?
                """, (json.dumps(context), datetime.now(), session_id))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating context: {e}")
    
    def update_preferences(self, session_id: str, preferences: Dict[str, Any]):
        """Update user preferences."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE conversations 
                    SET preferences = ?, last_updated = ?
                    WHERE session_id = ?
                """, (json.dumps(preferences), datetime.now(), session_id))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating preferences: {e}")
    
    def save_research_plan(self, session_id: str, research_plan: Dict[str, Any]):
        """Save a research plan."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO research_plans
                    (plan_id, session_id, query, steps, status, created_at, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    research_plan["plan_id"],
                    session_id,
                    research_plan["query"],
                    json.dumps(research_plan["steps"]),
                    research_plan["status"],
                    research_plan["created_at"],
                    research_plan.get("completed_at")
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving research plan: {e}")
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get session context and preferences."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT context, preferences, created_at, last_updated
                    FROM conversations 
                    WHERE session_id = ?
                """, (session_id,))
                
                row = cursor.fetchone()
                if row:
                    context_json, preferences_json, created_at, last_updated = row
                    return {
                        "context": json.loads(context_json) if context_json else {},
                        "preferences": json.loads(preferences_json) if preferences_json else {},
                        "created_at": created_at,
                        "last_updated": last_updated
                    }
                
                return {}
                
        except Exception as e:
            logger.error(f"Error getting session context: {e}")
            return {}
    
    def cleanup_old_sessions(self, days: int = 30):
        """Clean up old conversation sessions."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old messages
                cursor.execute("""
                    DELETE FROM messages 
                    WHERE session_id IN (
                        SELECT session_id FROM conversations 
                        WHERE last_updated < ?
                    )
                """, (cutoff_date,))
                
                # Delete old research plans
                cursor.execute("""
                    DELETE FROM research_plans 
                    WHERE session_id IN (
                        SELECT session_id FROM conversations 
                        WHERE last_updated < ?
                    )
                """, (cutoff_date,))
                
                # Delete old conversations
                cursor.execute("""
                    DELETE FROM conversations 
                    WHERE last_updated < ?
                """, (cutoff_date,))
                
                conn.commit()
                
            logger.info(f"Cleaned up sessions older than {days} days")
            
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {e}")


# Global memory manager instance
memory_manager = ConversationMemoryManager()