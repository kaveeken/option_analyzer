"""
Application entry point.

Run with: python -m option_analyzer
"""

import uvicorn

from .api.app import create_app
from .config import get_settings


def main() -> None:
    """Start the FastAPI application server."""
    settings = get_settings()

    uvicorn.run(
        create_app(),
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
