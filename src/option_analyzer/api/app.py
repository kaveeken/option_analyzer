"""
FastAPI application factory.

Creates and configures the FastAPI application with middleware and routes.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .middleware import error_handler_middleware
from .routes import health


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI application instance

    Configuration:
        - CORS middleware for cross-origin requests
        - Error handling middleware for domain exceptions
        - Health check endpoint
        - Interactive API docs at /docs and /redoc
    """
    app = FastAPI(
        title="Option Analyzer API",
        description="API for analyzing option trading strategies using historical data",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware - allows requests from any origin
    # In production, restrict origins to specific domains
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # TODO: Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Error handling middleware
    app.middleware("http")(error_handler_middleware)

    # Register routes
    app.include_router(health.router)

    return app
