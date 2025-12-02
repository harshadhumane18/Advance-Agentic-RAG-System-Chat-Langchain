# src/agents/research_planner.py
"""Research planning and execution system."""

import uuid
from typing import List, Dict, Any
from datetime import datetime
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_google_genai import ChatGoogleGenerativeAI

from config.agentic_settings import agentic_settings
from agents.state import ResearchPlan, ResearchStep


class ResearchPlanResult(BaseModel):
    """Structured output for the research plan."""
    steps: List[str] = Field(description="List of 3-5 specific research steps.")


class ResearchPlanner:
    """Intelligent research planner for complex queries."""
    
    def __init__(self):
        """Initialize the research planner."""
        self.model = ChatGoogleGenerativeAI(
            model=agentic_settings.research_model.split("/")[-1],
            google_api_key=agentic_settings.google_api_key,
            temperature=0.2
        ).with_structured_output(ResearchPlanResult)
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for research planning."""
        return """You are an expert research planner for a LangChain documentation assistant. Your task is to break down complex user queries into a series of actionable research steps.

        For each query, create 3-5 specific research steps. The steps should:
        1. Are focused and actionable
        2. Cover different aspects of the query
        3. Build upon each other logically
        4. Can be answered through document retrieval
        5. Are specific to LangChain/LangGraph/LangSmith technologies

        Each step should be a clear, specific question that can be researched independently.
        Respond ONLY with a JSON object matching the ResearchPlanResult schema."""

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception)
    )
    async def create_research_plan(self, query: str, context: Dict[str, Any] = None) -> ResearchPlan:
        """Create a research plan for a complex query."""
        try:
            logger.info(f"Creating research plan for query: {query[:100]}...")
            
            # Prepare messages
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"Query: {query}\nContext: {context or {}}")
            ]
            
            # Get research steps
            response: ResearchPlanResult = await self.model.ainvoke(messages)
            steps = response.steps[:agentic_settings.max_research_steps]
            
            # Create research plan
            plan_id = str(uuid.uuid4())
            research_steps = []
            
            for i, step_description in enumerate(steps):
                step = ResearchStep(
                    step_id=f"{plan_id}_step_{i+1}",
                    description=step_description,
                    status="pending"
                )
                research_steps.append(step)
            
            research_plan = ResearchPlan(
                plan_id=plan_id,
                query=query,
                steps=research_steps,
                status="created"
            )
            
            logger.info(f"Created research plan with {len(steps)} steps")
            return research_plan
            
        except Exception as e:
            logger.error(f"Error creating research plan: {e}")
            # Fallback to simple plan
            return self._create_fallback_plan(query)
    
    def _create_fallback_plan(self, query: str) -> ResearchPlan:
        """Create a fallback research plan."""
        plan_id = str(uuid.uuid4())
        fallback_steps = [
            ResearchStep(
                step_id=f"{plan_id}_step_1",
                description=f"Research general information about: {query}",
                status="pending"
            )
        ]
        
        return ResearchPlan(
            plan_id=plan_id,
            query=query,
            steps=fallback_steps,
            status="created"
        )


# Global research planner instance
research_planner = ResearchPlanner()