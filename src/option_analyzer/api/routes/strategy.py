"""
Strategy management endpoints.

Provides endpoints for initializing and managing option trading strategies.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Response

from ...clients.ibkr import IBKRClient
from ...models.session import SessionState
from ...services.session import SessionService
from ...utils.exceptions import InvalidQuantityError, MixedExpirationError, ValidationError
from ..dependencies import get_current_session, get_ibkr_client, get_session_service_dep
from ..schemas import (
    AddPositionRequest,
    ModifyPositionRequest,
    PositionResponse,
    PositionsResponse,
    StrategyInitRequest,
    StrategyInitResponse,
)

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


@router.post("/positions", response_model=PositionsResponse)
async def add_position(
    request: AddPositionRequest,
    session: Annotated[SessionState, Depends(get_current_session)],
    ibkr: Annotated[IBKRClient, Depends(get_ibkr_client)],
) -> PositionsResponse:
    """
    Add an option position to the strategy.

    Args:
        request: Position details (conid and quantity)
        session: Current session with strategy data
        ibkr: IBKR client dependency

    Returns:
        Updated list of all positions

    Raises:
        401: No valid session
        400: Invalid quantity (zero), mixed expirations, or contract not found
        502: IBKR API error

    Example:
        POST /api/positions
        Body: {"conid": 123456, "quantity": 2}
        Response: {"positions": [...]}
    """
    # Validate quantity != 0
    if request.quantity == 0:
        raise InvalidQuantityError()

    # Get strategy from session
    strategy = session.data.get("strategy")
    if not strategy:
        raise ValidationError("No strategy initialized. Call /api/strategy/init first.")

    # Get option chain for target_date to find the contract
    stock_conid = strategy["stock_conid"]
    target_date = strategy["target_date"]
    chain = await ibkr.get_option_chain(stock_conid, target_date)

    # Find contract by conid in the chain
    contract = None
    for c in list(chain.calls) + list(chain.puts):
        if c.conid == request.conid:
            contract = c
            break

    if not contract:
        raise ValidationError(
            f"Contract {request.conid} not found in option chain for {target_date}",
            code="CONTRACT_NOT_FOUND",
        )

    # Check for mixed expirations
    positions = strategy.get("positions", [])
    if positions:
        # All positions must have same expiration as the new contract
        for pos in positions:
            if pos["expiration"] != str(contract.expiration):
                raise MixedExpirationError(
                    f"Cannot add contract with expiration {contract.expiration}. "
                    f"Strategy already has positions with expiration {pos['expiration']}"
                )

    # Add position to strategy
    position_data = {
        "conid": contract.conid,
        "strike": contract.strike,
        "right": contract.right,
        "expiration": str(contract.expiration),
        "quantity": request.quantity,
        "bid": contract.bid,
        "ask": contract.ask,
    }
    positions.append(position_data)
    strategy["positions"] = positions

    # Return all positions
    return PositionsResponse(
        positions=[PositionResponse(**pos) for pos in positions]
    )


@router.patch("/positions/{conid}", response_model=PositionsResponse)
async def modify_position(
    conid: int,
    request: ModifyPositionRequest,
    session: Annotated[SessionState, Depends(get_current_session)],
) -> PositionsResponse:
    """
    Modify an existing position's quantity.

    Args:
        conid: Contract ID of position to modify
        request: New quantity
        session: Current session with strategy data

    Returns:
        Updated list of all positions

    Raises:
        401: No valid session
        400: Invalid quantity (zero) or position not found

    Example:
        PATCH /api/positions/123456
        Body: {"quantity": 3}
        Response: {"positions": [...]}
    """
    # Validate quantity != 0
    if request.quantity == 0:
        raise InvalidQuantityError()

    # Get strategy from session
    strategy = session.data.get("strategy")
    if not strategy:
        raise ValidationError("No strategy initialized. Call /api/strategy/init first.")

    # Find and update position
    positions = strategy.get("positions", [])
    found = False
    for pos in positions:
        if pos["conid"] == conid:
            pos["quantity"] = request.quantity
            found = True
            break

    if not found:
        raise ValidationError(
            f"Position with conid {conid} not found",
            code="POSITION_NOT_FOUND",
        )

    # Return all positions
    return PositionsResponse(
        positions=[PositionResponse(**pos) for pos in positions]
    )


@router.delete("/positions/{conid}", response_model=PositionsResponse)
async def delete_position(
    conid: int,
    session: Annotated[SessionState, Depends(get_current_session)],
) -> PositionsResponse:
    """
    Remove a position from the strategy.

    Args:
        conid: Contract ID of position to remove
        session: Current session with strategy data

    Returns:
        Updated list of remaining positions

    Raises:
        401: No valid session
        400: Position not found

    Example:
        DELETE /api/positions/123456
        Response: {"positions": [...]}
    """
    # Get strategy from session
    strategy = session.data.get("strategy")
    if not strategy:
        raise ValidationError("No strategy initialized. Call /api/strategy/init first.")

    # Find and remove position
    positions = strategy.get("positions", [])
    original_length = len(positions)
    positions = [pos for pos in positions if pos["conid"] != conid]

    if len(positions) == original_length:
        raise ValidationError(
            f"Position with conid {conid} not found",
            code="POSITION_NOT_FOUND",
        )

    strategy["positions"] = positions

    # Return remaining positions
    return PositionsResponse(
        positions=[PositionResponse(**pos) for pos in positions]
    )
