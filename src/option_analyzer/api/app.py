"""
FastAPI application factory.

Creates and configures the FastAPI application with middleware and routes.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..config import get_settings
from ..services.session import get_session_service
from .middleware import error_handler_middleware
from .routes import health, stocks, strategy


# Background task control
_cleanup_task: asyncio.Task | None = None
_shutdown_event: asyncio.Event | None = None
_plot_executor: ThreadPoolExecutor | None = None


def get_plot_executor() -> ThreadPoolExecutor:
    """
    Get the global thread pool executor for matplotlib operations.

    Returns:
        ThreadPoolExecutor for running matplotlib operations

    Raises:
        RuntimeError: If called before application startup
    """
    if _plot_executor is None:
        raise RuntimeError("Plot executor not initialized. Application not started?")
    return _plot_executor


async def session_cleanup_task() -> None:
    """
    Background task to periodically clean up expired sessions.

    Runs every session_cleanup_interval seconds (configured in settings).
    """
    global _shutdown_event
    settings = get_settings()
    session_service = get_session_service(settings)

    while not _shutdown_event.is_set():
        try:
            # Wait for cleanup interval or shutdown signal
            await asyncio.wait_for(
                _shutdown_event.wait(),
                timeout=settings.session_cleanup_interval,
            )
            # If we get here, shutdown was signaled
            break
        except asyncio.TimeoutError:
            # Timeout means it's time to cleanup
            cleaned = session_service.cleanup_expired_sessions()
            if cleaned > 0:
                print(f"Cleaned up {cleaned} expired sessions")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup/shutdown tasks.

    Starts background session cleanup task on startup,
    stops it on shutdown.
    """
    global _cleanup_task, _shutdown_event, _plot_executor

    # Startup
    _shutdown_event = asyncio.Event()
    _cleanup_task = asyncio.create_task(session_cleanup_task())

    # Initialize plot executor for matplotlib operations
    # Use 2 worker threads to allow concurrent plot generation without blocking
    _plot_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="plot")

    # Ensure plots directory exists
    plots_dir = Path("static/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Plot directory initialized: {plots_dir.absolute()}")

    yield

    # Shutdown
    if _shutdown_event:
        _shutdown_event.set()
    if _cleanup_task:
        await _cleanup_task
    if _plot_executor:
        _plot_executor.shutdown(wait=True)


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI application instance

    Configuration:
        - CORS middleware for cross-origin requests
        - Error handling middleware for domain exceptions
        - Background session cleanup task
        - Health check endpoint
        - Interactive API docs at /docs and /redoc
    """
    app = FastAPI(
        title="Option Analyzer API",
        description="API for analyzing option trading strategies using historical data",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
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
    app.include_router(stocks.router)
    app.include_router(strategy.router)

    # Mount static files for serving generated plots
    app.mount("/static", StaticFiles(directory="static"), name="static")

    return app
