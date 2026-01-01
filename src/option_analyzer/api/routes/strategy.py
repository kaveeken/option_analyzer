"""
Strategy management endpoints.

Provides endpoints for initializing and managing option trading strategies.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Response

from ...clients.ibkr import IBKRClient
from ...services.session import SessionService
from ...utils.exceptions import ValidationError
from ..dependencies import get_ibkr_client, get_session_service_dep
from ..schemas import StrategyInitRequest, StrategyInitResponse

router = APIRouter(prefix="/api/strategy", tags=["strategy"])


@router.post("/init", response_model=StrategyInitResponse)
async def initialize_strategy(
    request: StrategyInitRequest,
    response: Response,
    ibkr: Annotated[IBKRClient, Depends(get_ibkr_client)],
    session_service: Annotated[SessionService, Depends(get_session_service_dep)],
) -> StrategyInitResponse:
    """
    Initialize a new trading strategy.

    Creates or updates a session with strategy data including the stock
    and automatically selected target expiration date (earliest available).

    Args:
        request: Strategy initialization request with symbol
        response: FastAPI response for setting cookies
        ibkr: IBKR client dependency
        session_service: Session service dependency

    Returns:
        Strategy initialization response with stock info and session ID

    Raises:
        404: Symbol not found
        400: No available option expirations for symbol
        502: IBKR API error

    Example:
        POST /api/strategy/init
        Body: {"symbol": "AAPL"}
        Response: {
            "symbol": "AAPL",
            "current_price": 150.25,
            "target_date": "JAN26",
            "available_expirations": ["JAN26", "FEB26", "MAR26"],
            "session_id": "abc123..."
        }
    """
    # Get stock info from IBKR
    stock = await ibkr.get_stock(request.symbol.upper())

    # Validate that stock has available option expirations
    if not stock.available_expirations:
        raise ValidationError(
            f"No option expirations available for symbol '{stock.symbol}'",
            code="NO_EXPIRATIONS_AVAILABLE",
        )

    # Select earliest expiration as target_date
    target_date = stock.available_expirations[0]

    # Create or get session
    session = session_service.create_session()

    # Store strategy data in session
    session.data["strategy"] = {
        "symbol": stock.symbol,
        "stock_conid": stock.conid,
        "current_price": stock.current_price,
        "target_date": target_date,
        "available_expirations": stock.available_expirations,
        "positions": [],  # Will be populated by position endpoints
    }

    # Set session cookie
    response.set_cookie(key="session_id", value=session.session_id)

    return StrategyInitResponse(
        symbol=stock.symbol,
        current_price=stock.current_price,
        target_date=target_date,
        available_expirations=stock.available_expirations,
        session_id=session.session_id,
    )
