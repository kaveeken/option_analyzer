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


class StrategyInitRequest(BaseModel):
    """
    Request to initialize a new strategy.

    Attributes:
        symbol: Stock ticker symbol
    """

    symbol: str = Field(
        description="Stock ticker symbol",
        examples=["AAPL"],
        min_length=1,
        max_length=10,
    )


class StrategyInitResponse(BaseModel):
    """
    Response from strategy initialization.

    Attributes:
        symbol: Stock ticker symbol
        current_price: Current stock price
        target_date: Automatically selected target expiration date
        available_expirations: All available expiration months
        session_id: Session ID for subsequent requests
    """

    symbol: str = Field(description="Stock ticker symbol")
    current_price: float = Field(description="Current stock price")
    target_date: str = Field(
        description="Automatically selected target expiration (earliest)",
        examples=["JAN26"],
    )
    available_expirations: list[str] = Field(
        description="All available expiration months"
    )
    session_id: str = Field(description="Session ID for subsequent requests")


class AddPositionRequest(BaseModel):
    """
    Request to add an option position to the strategy.

    Attributes:
        conid: IBKR contract identifier for the option
        quantity: Number of contracts (positive=long, negative=short, cannot be 0)
    """

    conid: int = Field(description="IBKR contract identifier", examples=[123456])
    quantity: int = Field(
        description="Number of contracts (positive=long, negative=short)",
        examples=[1, -2],
    )


class ModifyPositionRequest(BaseModel):
    """
    Request to modify an existing position's quantity.

    Attributes:
        quantity: New quantity (positive=long, negative=short, cannot be 0)
    """

    quantity: int = Field(
        description="New quantity (positive=long, negative=short)",
        examples=[2, -1],
    )


class PositionResponse(BaseModel):
    """
    Option position details.

    Attributes:
        conid: IBKR contract identifier
        strike: Strike price
        right: Option type (C=call, P=put)
        expiration: Expiration date
        quantity: Number of contracts
        bid: Bid price per share
        ask: Ask price per share
    """

    conid: int = Field(description="IBKR contract identifier")
    strike: float = Field(description="Strike price")
    right: str = Field(description="Option type", examples=["C", "P"])
    expiration: date = Field(description="Expiration date")
    quantity: int = Field(description="Number of contracts")
    bid: float | None = Field(description="Bid price per share")
    ask: float | None = Field(description="Ask price per share")


class PositionsResponse(BaseModel):
    """
    List of all positions in the strategy.

    Attributes:
        positions: List of option positions
    """

    positions: list[PositionResponse] = Field(description="List of option positions")


class PriceBinResponse(BaseModel):
    """
    A histogram bin representing a price range and frequency.

    Attributes:
        lower: Lower bound of the price range
        upper: Upper bound of the price range
        count: Number of simulated outcomes in this range
        midpoint: Midpoint of the price range
    """

    lower: float = Field(description="Lower bound of price range", examples=[145.0])
    upper: float = Field(description="Upper bound of price range", examples=[150.0])
    count: int = Field(description="Number of outcomes in this bin", examples=[127])
    midpoint: float = Field(description="Midpoint of the bin", examples=[147.5])


class StrategyAnalysisResponse(BaseModel):
    """
    Strategy analysis results including Monte Carlo simulation.

    Attributes:
        price_distribution: Histogram bins of simulated price outcomes
        expected_value: Probability-weighted average P&L in dollars
        probability_of_profit: Fraction of outcomes with positive P&L (0.0 to 1.0)
        max_gain: Maximum possible profit (None if unlimited upside)
        max_loss: Maximum possible loss (None if unlimited downside)
    """

    price_distribution: list[PriceBinResponse] = Field(
        description="Histogram of simulated price outcomes"
    )
    expected_value: float = Field(
        description="Probability-weighted average P&L", examples=[250.50]
    )
    probability_of_profit: float = Field(
        description="Fraction of profitable outcomes", ge=0.0, le=1.0, examples=[0.68]
    )
    max_gain: float | None = Field(
        description="Maximum possible profit (None if unlimited)", examples=[1000.0]
    )
    max_loss: float | None = Field(
        description="Maximum possible loss (None if unlimited)", examples=[-500.0]
    )
