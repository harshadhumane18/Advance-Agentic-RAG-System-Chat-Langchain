# src/main.py (Updated)
"""Main entry point for the agentic RAG system."""

import argparse
import sys
import asyncio
import uuid
from pathlib import Path
from loguru import logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from ingestion.orchestrator import run_ingestion_pipeline
from agents.agentic_orchestrator import run_with_graph
from agents.state import InputState


async def run_agentic_query(query: str, session_id: str = None, streaming: bool = False) -> str:
    """Run a single agentic query via LangGraph."""
    if not session_id:
        session_id = f"session_{hash(query) % 10000}"
    
    input_state = InputState(
        query=query,
        session_id=session_id,
        streaming=streaming
    )
    
    result = await run_with_graph(input_state)
    return result.generated_response


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Agentic RAG System")
    parser.add_argument("--mode", choices=["ingest", "query", "interactive"], 
                       default="interactive", help="Operation mode")
    parser.add_argument("--query", type=str, help="Query to process")
    parser.add_argument("--session-id", type=str, help="Session ID")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming")
    parser.add_argument("--force-update", action="store_true", 
                       help="Force update ingestion")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Set the logging level")
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(
        "logs/agentic_rag.log",
        level=args.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="10 MB"
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level=args.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}"
    )
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    try:
        if args.mode == "ingest":
            logger.info("Starting ingestion pipeline")
            stats = run_ingestion_pipeline(force_update=args.force_update)
            logger.info("Ingestion completed successfully")
            logger.info(f"Statistics: {stats}")
            
        elif args.mode == "query":
            if not args.query:
                print("Error: --query is required for query mode")
                return
            
            logger.info(f"Processing query: {args.query}")
            response = asyncio.run(run_agentic_query(args.query, args.session_id, args.streaming))
            print(f"🤖 Assistant: {response}")
            
        elif args.mode == "interactive":
            print("🤖 Agentic RAG System - Interactive Mode")
            print("Type 'quit' to exit, 'ingest' to run ingestion")
            print("-" * 50)
            
            session_id = args.session_id or f"session_{uuid.uuid4().hex[:8]}"
            
            while True:
                try:
                    query = input("\n👤 You: ").strip()
                    
                    if query.lower() in ['quit', 'exit', 'q']:
                        print("👋 Goodbye!")
                        break
                    elif query.lower() == 'ingest':
                        print("🔄 Running ingestion...")
                        stats = run_ingestion_pipeline(force_update=args.force_update)
                        print(f"✅ Ingestion completed: {stats}")
                        continue
                    elif not query:
                        continue
                    
                    response = asyncio.run(run_agentic_query(query, session_id, args.streaming))
                    print(f"🤖 Assistant: {response}")
                    
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    logger.error(f"Error in interactive mode: {e}")
                    print(f"❌ Error: {e}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()