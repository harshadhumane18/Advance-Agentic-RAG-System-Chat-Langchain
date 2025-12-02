# scripts/start_api.py
"""Start the API server."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import uvicorn
from api.server import app
from config.agentic_settings import agentic_settings

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=agentic_settings.api_host,
        port=agentic_settings.api_port,
        reload=True,
        log_level="info"
    )
    