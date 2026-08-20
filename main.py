"""Entry point: `python main.py` starts the retrieval API with uvicorn."""

import uvicorn

from src.config import settings

if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host=settings.api_host, port=settings.api_port,reload=True)
