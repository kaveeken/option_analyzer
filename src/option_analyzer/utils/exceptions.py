"""
Custom exceptions for the option analyzer application.

These exceptions provide structured error handling for different failure scenarios.
"""


class OptionAnalyzerError(Exception):
    """Base exception for all option analyzer errors."""

    def __init__(self, message: str, code: str = "UNKNOWN_ERROR") -> None:
        """
        Initialize exception with message and error code.

        Args:
            message: Human-readable error message
            code: Machine-readable error code (SNAKE_CASE)
        """
        self.message = message
        self.code = code
        super().__init__(message)


class IBKRAPIError(OptionAnalyzerError):
    """Raised when IBKR API request fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="IBKR_API_ERROR")


class ValidationError(OptionAnalyzerError):
    """Raised when data validation fails."""

    def __init__(self, message: str, code: str = "VALIDATION_ERROR") -> None:
        super().__init__(message, code=code)


class MixedExpirationError(ValidationError):
    """Raised when strategy contains options with different expiration dates."""

    def __init__(self, message: str = "Strategy contains options with different expiration dates") -> None:
        super().__init__(message, code="MIXED_EXPIRATION")


class InvalidQuantityError(ValidationError):
    """Raised when quantity is zero or invalid."""

    def __init__(self, message: str = "Quantity must be non-zero") -> None:
        super().__init__(message, code="INVALID_QUANTITY")


class MissingBidAskError(ValidationError):
    """Raised when option has no bid/ask data available."""

    def __init__(self, message: str = "Option has no bid/ask data available") -> None:
        super().__init__(message, code="MISSING_BID_ASK")


class SymbolNotFoundError(OptionAnalyzerError):
    """Raised when stock symbol is not found."""

    def __init__(self, symbol: str) -> None:
        message = f"Symbol '{symbol}' not found"
        super().__init__(message, code="SYMBOL_NOT_FOUND")


class RateLimitedError(OptionAnalyzerError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str = "Too many requests to IBKR API") -> None:
        super().__init__(message, code="RATE_LIMITED")


class SessionExpiredError(OptionAnalyzerError):
    """Raised when session is not found or expired."""

    def __init__(self, message: str = "Session not found or expired") -> None:
        super().__init__(message, code="SESSION_EXPIRED")
