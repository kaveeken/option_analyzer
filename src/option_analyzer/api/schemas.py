"""
API request and response schemas.

These Pydantic models define the contract between the API and clients.
"""

from datetime import date

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """
    Standardized error response format.

    Attributes:
        error: Human-readable error message
        code: Machine-readable error code (SNAKE_CASE)
        details: Optional additional context
    """

    error: str = Field(description="Human-readable error message")
    code: str = Field(description="Machine-readable error code")
    details: dict[str, str] | None = Field(
        default=None, description="Optional additional error context"
    )


class HealthCheckResponse(BaseModel):
    """
    Health check endpoint response.

    Attributes:
        status: Current service status
        version: API version
    """

    status: str = Field(description="Service status", examples=["healthy"])
    version: str = Field(description="API version", examples=["0.1.0"])


class StockResponse(BaseModel):
    """
    Stock information response.

    Attributes:
        symbol: Stock ticker symbol
        current_price: Current market price
        conid: IBKR contract identifier
        available_expirations: Available option expiration months
    """

    symbol: str = Field(description="Stock ticker symbol", examples=["AAPL"])
    current_price: float = Field(description="Current market price", examples=[150.25])
    conid: int = Field(description="IBKR contract identifier")
    available_expirations: list[str] = Field(
        description="Available option expiration months", examples=[["JAN26", "FEB26"]]
    )


class OptionContractResponse(BaseModel):
    """
    Option contract details.

    Attributes:
        conid: IBKR contract identifier
        strike: Strike price
        right: Option type (C=call, P=put)
        expiration: Expiration date
        bid: Current bid price per share
        ask: Current ask price per share
        multiplier: Shares per contract
    """

    conid: int = Field(description="IBKR contract identifier")
    strike: float = Field(description="Strike price", examples=[150.0])
    right: str = Field(description="Option type", examples=["C", "P"])
    expiration: date = Field(description="Expiration date")
    bid: float | None = Field(description="Bid price per share", examples=[2.50])
    ask: float | None = Field(description="Ask price per share", examples=[2.55])
    multiplier: int = Field(description="Shares per contract", examples=[100])


class OptionChainResponse(BaseModel):
    """
    Option chain for a specific expiration.

    Attributes:
        expiration: Option expiration date
        calls: List of call option contracts
        puts: List of put option contracts
    """

    expiration: date = Field(description="Option expiration date")
    calls: list[OptionContractResponse] = Field(description="Call option contracts")
    puts: list[OptionContractResponse] = Field(description="Put option contracts")
