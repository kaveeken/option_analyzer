# Options Analysis Platform - Architecture & Implementation Plan

## Overview

A web-based platform for analyzing option trading strategies using historical price distributions and Monte Carlo simulation. Users can visualize expected returns across different market scenarios and optimize multi-leg option positions.

## Core Capabilities

1. **Historical Price Analysis**: Fetch and analyze historical stock prices to model future price distributions
2. **Options Chain Visualization**: Display real-time options data with interactive position building
3. **Strategy Modeling**: Simulate P&L for complex multi-leg option strategies (calls, puts, underlying)
4. **Monte Carlo Simulation**: Bootstrap historical returns to generate probabilistic future scenarios
5. **Risk Metrics**: Calculate expected value, probability of profit, and risk/reward ratios

## Technical Stack

**Backend**:
- FastAPI (async API framework)
- Pydantic v2 (data validation, settings)
- HTTPX (async HTTP client for IBKR)
- NumPy (vectorized statistics)
- Matplotlib (plotting)

**Frontend**:
- Static HTML/CSS
- Modern JavaScript (ES6 modules)
- No framework dependencies

**Infrastructure**:
- Uvicorn (ASGI server)
- In-memory cache (Redis-ready interface)
- IBKR Client Portal API

## Project Structure

```
option-analyzer/
├── pyproject.toml
├── .env.example
│
├── src/
│   └── option_analyzer/
│       ├── main.py                    # FastAPI app + startup
│       ├── config.py                  # Environment config
│       │
│       ├── models/
│       │   ├── domain.py             # Core business models
│       │   ├── api.py                # API request/response schemas
│       │   └── ibkr.py               # IBKR API response schemas
│       │
│       ├── services/
│       │   ├── market_data.py        # IBKR client + caching
│       │   ├── statistics.py         # Price distribution analysis
│       │   ├── strategy.py           # Option strategy P&L modeling
│       │   ├── risk.py               # Risk metrics calculation
│       │   ├── plotting.py           # Chart generation
│       │   └── session.py            # Session state management
│       │
│       ├── api/
│       │   ├── dependencies.py       # DI, session management, cookies
│       │   └── routes/
│       │       ├── market.py         # /stock, /chain endpoints
│       │       ├── strategy.py       # /position, /calculate endpoints
│       │       └── analysis.py       # /plot, /metrics endpoints
│       │
│       ├── cache/
│       │   ├── interface.py          # Cache protocol
│       │   └── memory.py             # In-memory implementation
│       │
│       └── utils/
│           ├── exceptions.py
│           ├── logging.py
│           └── rate_limiter.py
│
├── static/
│   ├── index.html
│   ├── css/
│   │   └── styles.css
│   ├── js/
│   │   ├── api.js                    # Backend API client
│   │   ├── ui.js                     # DOM manipulation
│   │   ├── state.js                  # Frontend state management
│   │   ├── chart.js                  # Chart display logic
│   │   └── main.js                   # App controller
│   └── plots/                        # Per-session generated charts
│       └── {session_id}_{timestamp}.png
│
└── tests/
    ├── conftest.py
    ├── unit/
    │   ├── test_statistics.py
    │   ├── test_strategy.py
    │   ├── test_risk.py
    │   └── test_rate_limiter.py
    ├── integration/
    │   ├── test_api_workflow.py
    │   └── test_ibkr_real.py        # Optional: real IBKR integration
    ├── e2e/
    │   └── test_browser_flow.py     # Playwright tests
    ├── performance/
    │   └── test_benchmarks.py
    └── fixtures/
        └── sample_data.json
```

## Domain Model

### Core Abstractions

All tradeable instruments implement a common interface:

```python
class Tradeable(BaseModel):
    """Base class for anything that has a payoff function"""

    def payoff_at_price(self, price: float) -> float:
        """Calculate the value of this position at a given underlying price"""
        raise NotImplementedError
```

### Models

**Stock Position**
```python
from datetime import date

class Stock(BaseModel):
    symbol: str
    current_price: float
    conid: str
    available_expirations: List[date]  # Option expiration dates

    def payoff_at_price(self, price: float) -> float:
        """Linear payoff: gain/loss per share"""
        return price - self.current_price
```

**Option Contract**
```python
from datetime import date

class OptionContract(BaseModel):
    conid: str
    strike: float
    right: Literal["C", "P"]
    expiration: date  # Actual expiration date (e.g., 2024-12-20)
    bid: Optional[float]
    ask: Optional[float]
    multiplier: int = 100  # Shares per contract (usually 100)

    def intrinsic_value(self, price: float) -> float:
        """Value at expiration (no time value)"""
        if self.right == "C":
            return max(0, price - self.strike) * self.multiplier
        else:
            return max(0, self.strike - price) * self.multiplier

    def days_to_expiry(self, reference_date: Optional[date] = None) -> int:
        """Calculate trading days until expiration"""
        ref = reference_date or date.today()
        return (self.expiration - ref).days
```

**Option Position**
```python
class OptionPosition(BaseModel):
    """A position in a specific option contract"""
    contract: OptionContract
    quantity: int  # positive = long, negative = short

    @property
    def premium_paid(self) -> float:
        """Upfront cost/credit for this position (total, not per share)"""
        if self.quantity > 0:
            # Long: pay ask (total cost = quantity * ask * multiplier)
            price = self.contract.ask if self.contract.ask is not None else 0
            return -self.quantity * price * self.contract.multiplier
        else:
            # Short: receive bid (total credit = quantity * bid * multiplier)
            price = self.contract.bid if self.contract.bid is not None else 0
            return -self.quantity * price * self.contract.multiplier

    def payoff_at_price(self, price: float) -> float:
        """Total P&L including premium and intrinsic value"""
        intrinsic = self.quantity * self.contract.intrinsic_value(price)
        return self.premium_paid + intrinsic
```

**Strategy** (Portfolio)
```python
class Strategy(BaseModel):
    """A complete trading strategy with stock + options"""
    stock: Stock
    stock_quantity: int = 0  # Shares held
    option_positions: List[OptionPosition] = []

    def total_payoff(self, price: float, include_transaction_costs: bool = True) -> float:
        """Aggregate P&L across all positions"""
        stock_pnl = self.stock_quantity * (price - self.stock.current_price)
        options_pnl = sum(pos.payoff_at_price(price) for pos in self.option_positions)
        return stock_pnl + options_pnl

    @property
    def net_premium(self) -> float:
        """Total upfront cost (negative) or credit (positive)"""
        return sum(pos.premium_paid for pos in self.option_positions)

    @property
    def transaction_costs(self) -> float:
        """Total commission costs for all option positions"""
        from ..config import get_settings
        cost_per_contract = get_settings().transaction_cost_per_contract
        return sum(abs(pos.quantity) * cost_per_contract for pos in self.option_positions)

    @property
    def max_risk(self) -> Optional[float]:
        """Maximum possible loss (None if unlimited)"""
        # Analyze the payoff function across all prices
        ...

    def get_earliest_expiration(self) -> Optional[date]:
        """Get the earliest expiration date from all option positions"""
        if not self.option_positions:
            return None
        return min(pos.contract.expiration for pos in self.option_positions)

    def validate_single_expiration(self) -> bool:
        """Check that all options have the same expiration (required for analysis)"""
        if not self.option_positions:
            return True
        expirations = {pos.contract.expiration for pos in self.option_positions}
        return len(expirations) == 1
```

## Statistical Engine

### Price Distribution Modeling

**Geometric Returns**
```python
def geometric_returns(closes: np.ndarray, period: int = 1) -> np.ndarray:
    """
    Calculate period-over-period price multipliers.

    Returns: closes[t+period] / closes[t]
    Example: If prices go [100, 110, 121], returns are [1.10, 1.10]
    """
    return closes[period:] / closes[:-period]

def calculate_period_for_days(bar_size: str, target_days: int) -> int:
    """
    Convert target days to number of periods based on bar size.

    Args:
        bar_size: IBKR bar size (e.g., "1d", "1w", "1m")
        target_days: Calendar days to expiration

    Returns:
        Number of periods (bars) to simulate

    Example:
        - bar_size="1w", target_days=30 -> 4-5 weeks
        - bar_size="1d", target_days=30 -> ~21 trading days
    """
    days_per_bar = {
        "1d": 1,
        "1w": 7,
        "1m": 30,
    }

    bar_days = days_per_bar.get(bar_size, 1)
    # Adjust for trading days (~252 trading days / 365 calendar days)
    trading_day_ratio = 252 / 365

    return int((target_days * trading_day_ratio) / (bar_days * trading_day_ratio))
```

**Bootstrap Simulation**
```python
def bootstrap_walk(returns: np.ndarray, steps: int) -> float:
    """
    Simulate one random walk by sampling historical returns.

    Example: If returns = [1.1, 0.95, 1.05] and steps = 3:
      - Sample randomly with replacement 3 times
      - Multiply together: maybe 1.1 × 1.05 × 0.95 = 1.0973
    """
    samples = np.random.choice(returns, size=steps, replace=True)
    return np.prod(samples)

def monte_carlo_simulation(
    returns: np.ndarray,
    target_date: date,
    bar_size: str = "1w",
    n_simulations: int = 10000,
    reference_date: Optional[date] = None
) -> np.ndarray:
    """
    Run bootstrap simulations to target date.

    Args:
        returns: Historical geometric returns
        target_date: Date to simulate to (e.g., option expiration)
        bar_size: Historical bar size used (1d, 1w, etc.)
        n_simulations: Number of Monte Carlo paths
        reference_date: Starting date (default: today)

    Returns:
        Array of final price multipliers
    """
    ref = reference_date or date.today()
    days_to_target = (target_date - ref).days
    steps = calculate_period_for_days(bar_size, days_to_target)

    return np.array([bootstrap_walk(returns, steps) for _ in range(n_simulations)])

def get_price_distribution(
    current_price: float,
    returns: np.ndarray,
    target_date: date,
    bootstrap_samples: Optional[int],
    bar_size: str = "1w"
) -> np.ndarray:
    """
    Generate price distribution for analysis.

    Args:
        current_price: Current stock price
        returns: Historical geometric returns
        target_date: Date to analyze
        bootstrap_samples: If None, use direct returns; if >0, use Monte Carlo
        bar_size: Historical bar size

    Returns:
        Array of future prices (not multipliers)
    """
    if bootstrap_samples is None:
        # Direct historical returns: use actual return sequence
        multipliers = returns
    else:
        # Monte Carlo: bootstrap random walks
        multipliers = monte_carlo_simulation(
            returns, target_date, bar_size, bootstrap_samples
        )

    return current_price * multipliers

def calculate_price_range(
    price_distribution: np.ndarray,
    user_min: Optional[float] = None,
    user_max: Optional[float] = None
) -> Tuple[float, float]:
    """
    Calculate price range for plotting and analysis.

    Args:
        price_distribution: Array of simulated prices
        user_min: User-defined minimum (None = auto)
        user_max: User-defined maximum (None = auto)

    Returns:
        (min_price, max_price) tuple

    Default: Use bootstrap distribution min/max with 5% padding
    """
    if user_min is None:
        min_price = price_distribution.min() * 0.95
    else:
        min_price = user_min

    if user_max is None:
        max_price = price_distribution.max() * 1.05
    else:
        max_price = user_max

    return min_price, max_price
```

**Histogram Binning**
```python
class Bin(BaseModel):
    lower: float
    upper: float
    count: int

    @property
    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2

def create_histogram(
    values: np.ndarray,
    n_bins: int
) -> List[Bin]:
    """
    Create equal-width bins ensuring max value is included.

    Edge case: Last bin upper bound is max(values) + ε to include maximum.
    """
    min_val, max_val = values.min(), values.max()
    bin_width = (max_val - min_val) / n_bins

    bins = []
    for i in range(n_bins):
        lower = min_val + i * bin_width
        # Ensure last bin captures maximum value
        upper = max_val + 1e-10 if i == n_bins - 1 else lower + bin_width
        count = np.sum((values >= lower) & (values < upper))
        bins.append(Bin(lower=lower, upper=upper, count=count))

    return bins
```

### Risk Metrics

**Expected Value**
```python
def calculate_expected_value(
    strategy: Strategy,
    price_distribution: List[Bin]
) -> float:
    """
    Probability-weighted average return.

    For each bin:
      - Calculate P&L at bin midpoint
      - Weight by bin probability (count / total_count)
      - Sum weighted P&Ls
    """
    total_count = sum(bin.count for bin in price_distribution)

    expected_value = sum(
        (bin.count / total_count) * strategy.total_payoff(bin.midpoint)
        for bin in price_distribution
    )

    return expected_value
```

**Probability of Profit**
```python
def probability_of_profit(
    strategy: Strategy,
    price_distribution: List[Bin]
) -> float:
    """Fraction of scenarios where strategy makes money"""
    total_count = sum(bin.count for bin in price_distribution)
    profitable_count = sum(
        bin.count for bin in price_distribution
        if strategy.total_payoff(bin.midpoint) > 0
    )
    return profitable_count / total_count
```

**Risk Metrics**
```python
class RiskMetrics(BaseModel):
    expected_value: float
    probability_of_profit: float
    max_gain: Optional[float]  # None if unlimited
    max_loss: Optional[float]  # None if unlimited
    breakeven_prices: List[float]
    sharpe_ratio: Optional[float]  # If risk-free rate provided
```

## IBKR Integration

### Client Architecture

```python
class IBKRClient:
    """Async client for IBKR Client Portal API with rate limiting"""

    def __init__(self, base_url: str, cache: CacheInterface):
        self.base_url = base_url  # https://localhost:5000/v1/api
        self.cache = cache
        self.client = httpx.AsyncClient(verify=False, timeout=30.0)
        self._rate_limiter = RateLimiter(max_requests=50, per_seconds=60)

    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        max_retries: int = 3,
        **kwargs
    ) -> httpx.Response:
        """
        Make HTTP request with rate limiting and retry logic.

        Raises:
            IBKRAPIError: If all retries exhausted
        """
        await self._rate_limiter.acquire()

        for attempt in range(max_retries):
            try:
                response = await self.client.request(method, f"{self.base_url}/{endpoint}", **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limited
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                elif e.response.status_code >= 500:  # Server error
                    await asyncio.sleep(1)
                    continue
                else:
                    raise IBKRAPIError(f"IBKR API error: {e}") from e
            except httpx.RequestError as e:
                if attempt == max_retries - 1:
                    raise IBKRAPIError(f"IBKR connection failed: {e}") from e
                await asyncio.sleep(1)

        raise IBKRAPIError("Max retries exceeded")

    async def get_stock(self, symbol: str) -> Stock:
        """
        Workflow:
        1. Search for symbol (secdef/search)
        2. Get current price (marketdata/snapshot)
        3. Extract available option months
        """
        ...

    async def get_option_chain(
        self,
        stock: Stock,
        expiration: date
    ) -> List[OptionContract]:
        """
        Workflow:
        1. Check cache (key: f"{symbol}:{expiration.isoformat()}")
        2. Get strikes for expiration date (secdef/strikes)
        3. Fetch bid/ask for each strike (marketdata/snapshot)
        4. Parse expiration date from IBKR response
        5. Cache for 5 minutes

        Note: IBKR returns expiration as YYYYMMDD string, convert to date object
        """
        ...

    async def get_historical_prices(
        self,
        conid: str,
        years: int = 5,
        bar: str = "1w"
    ) -> List[float]:
        """
        Fetch OHLC data and extract close prices.

        Args:
            conid: IBKR contract ID
            years: Number of years of history to fetch
            bar: Bar size (1d, 1w, 1m)

        Returns:
            Array of closing prices (oldest to newest)

        Note: IBKR has different period formats:
            - "5y" for 5 years
            - "12m" for 12 months
            - Adjust based on requested years
        """
        ...
```

### Caching Strategy

```python
class CacheInterface(Protocol):
    async def get(self, key: str) -> Optional[Any]: ...
    async def set(self, key: str, value: Any, ttl: Optional[int] = None): ...
    async def delete(self, key: str): ...

# Cache TTLs
CACHE_TTLS = {
    "stock_price": 60,        # 1 minute
    "option_chain": 300,      # 5 minutes
    "historical": 86400,      # 24 hours (doesn't change)
}
```

## API Design

### Session State

```python
from datetime import date
from uuid import uuid4

class AnalysisParameters(BaseModel):
    """Analysis configuration for strategy evaluation"""
    target_date: Optional[date] = None  # Auto-set to earliest option expiration if None
    bin_count: int = 50
    bootstrap_samples: Optional[int] = None  # None = direct historical returns, >0 = Monte Carlo
    historical_years: int = 5  # Years of historical data to use for analysis
    price_range_min: Optional[float] = None  # Min price for plotting (None = auto from bootstrap)
    price_range_max: Optional[float] = None  # Max price for plotting (None = auto from bootstrap)
    transaction_cost_per_contract: float = 0.65  # Commission per contract

class SessionState(BaseModel):
    """
    Per-user session state for strategy building.

    Each browser session gets a unique session_id stored in a cookie.
    """
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    current_stock: Optional[Stock] = None
    current_strategy: Optional[Strategy] = None
    analysis_params: AnalysisParameters = Field(default_factory=AnalysisParameters)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)

    def touch(self):
        """Update last accessed timestamp"""
        self.last_accessed = datetime.utcnow()

# Global session storage (in-memory, upgradeable to Redis)
_sessions: Dict[str, SessionState] = {}
SESSION_TTL = 3600  # 1 hour

def get_or_create_session(session_id: Optional[str] = None) -> SessionState:
    """Get existing session or create new one"""
    if session_id and session_id in _sessions:
        session = _sessions[session_id]
        session.touch()
        return session

    # Create new session
    session = SessionState()
    _sessions[session.session_id] = session
    return session

def cleanup_expired_sessions():
    """Remove sessions older than SESSION_TTL (called by background task)"""
    now = datetime.utcnow()
    expired = [
        sid for sid, sess in _sessions.items()
        if (now - sess.last_accessed).total_seconds() > SESSION_TTL
    ]
    for sid in expired:
        del _sessions[sid]

    # Also cleanup old plot files
    import os
    from pathlib import Path
    plots_dir = Path("static/plots")
    if plots_dir.exists():
        for plot_file in plots_dir.glob("*.png"):
            if (now.timestamp() - plot_file.stat().st_mtime) > SESSION_TTL:
                plot_file.unlink()

# Background task setup (in main.py)
async def session_cleanup_task():
    """Background task that runs every 5 minutes"""
    while True:
        await asyncio.sleep(300)  # 5 minutes
        cleanup_expired_sessions()
```

### Endpoints

All endpoints use session cookies for state management:
- `Set-Cookie: session_id=<uuid>; Path=/; HttpOnly; SameSite=Lax`
- Session auto-created on first request
- Sessions expire after 1 hour of inactivity

**Market Data**
```
GET /api/v1/stock?symbol=AAPL
Response: {
  symbol,
  currentPrice,
  conid,
  availableExpirations: ["2024-12-20", "2025-01-17", ...]
}

GET /api/v1/chain?symbol=AAPL&expiration=2024-12-20
Response: {
  expiration: "2024-12-20",
  daysToExpiry: 30,
  strikes: [
    {
      strike,
      call: { conid, bid, ask, multiplier: 100, expiration: "2024-12-20" },
      put: { conid, bid, ask, multiplier: 100, expiration: "2024-12-20" }
    }
  ]
}

GET /api/v1/history?symbol=AAPL&period=12y&bar=1w
Response: { closes: [100.5, 102.3, ...] }
```

**Strategy Building**
```
POST /api/v1/strategy/stock
Body: { symbol, quantity }
Response: { success, netPosition, sessionId }

POST /api/v1/strategy/option
Body: { conid, strike, right: "C"|"P", expiration: "2024-12-20", quantity }
Response: { success, netPremium, totalContracts, sessionId }

PATCH /api/v1/strategy/option/{index}
Body: { quantity }
Response: { success, sessionId }

DELETE /api/v1/strategy/option/{index}
Response: { success, sessionId }

GET /api/v1/strategy/current
Response: {
  stock,
  stockQuantity,
  optionPositions: [
    { contract: { strike, right, expiration, multiplier, ... }, quantity }
  ],
  netPremium,
  sessionId
}

GET /api/v1/strategy/summary
Response: {
  totalCapitalRequired,
  maxProfit,      # None if unlimited
  maxLoss,        # None if unlimited
  netPremium,
  transactionCosts,
  positionCount,
  sessionId
}

POST /api/v1/strategy/reset
Response: { success, sessionId }
```

**Analysis**
```
POST /api/v1/analyze/plot
Body: {
  targetDate?: "2024-12-20",  # Optional: auto-set to earliest option expiration if null
  binCount?: 50,
  bootstrapSamples?: null,     # null = direct returns, >0 = Monte Carlo
  historicalYears?: 5,
  priceRangeMin?: null,        # null = auto from bootstrap
  priceRangeMax?: null
}
Response: {
  plotUrl: "/static/plots/{session_id}_{timestamp}.png",
  timestamp,
  daysToTarget: 30,
  targetDate: "2024-12-20",    # The actual date used (after auto-select)
  sessionId
}

GET /api/v1/analyze/metrics
Query: ?targetDate=2024-12-20
Response: {
  expectedValue,
  probabilityOfProfit,
  maxGain,
  maxLoss,
  breakevenPrices[],
  daysToTarget: 30,
  sessionId
}

POST /api/v1/analyze/parameters
Body: {
  targetDate?: "2024-12-20",
  binCount?: 50,
  bootstrapSamples?: null,
  historicalYears?: 5,
  priceRangeMin?: null,
  priceRangeMax?: null,
  transactionCostPerContract?: 0.65
}
Response: {
  targetDate,
  binCount,
  bootstrapSamples,
  historicalYears,
  priceRangeMin,
  priceRangeMax,
  transactionCostPerContract,
  sessionId
}
```

**Validation & Error Responses**

All endpoints return standardized errors:
```
{
  "error": "Human-readable message",
  "code": "SNAKE_CASE_ERROR_CODE",
  "details": { ... }  # Optional context
}
```

Common validation errors:
- `MIXED_EXPIRATION`: Strategy contains options with different expiration dates
- `INVALID_QUANTITY`: Quantity is zero or invalid
- `MISSING_BID_ASK`: Option has no bid/ask data available
- `IBKR_API_DOWN`: Cannot connect to IBKR API
- `SYMBOL_NOT_FOUND`: Stock symbol not found
- `RATE_LIMITED`: Too many requests to IBKR
- `SESSION_EXPIRED`: Session not found or expired
```

**Health**
```
GET /health
Response: { status: "healthy", ibkr: "connected" }
```

## Visualization

### Dual-Axis Chart

**Left Y-axis**: Histogram of price distribution (count)
**Right Y-axis**: Strategy P&L ($)
**X-axis**: Underlying price

**Elements**:
1. **Histogram bars**: Showing probable future prices
2. **P&L curve**: Strategy payoff across price range
3. **Current price line**: Vertical line at current stock price
4. **Zero line**: Horizontal line at $0 P&L
5. **Breakeven markers**: Points where P&L crosses zero

```python
async def generate_strategy_plot(
    strategy: Strategy,
    price_distribution: List[Bin],
    current_price: float,
    session_id: str
) -> str:
    """
    Create matplotlib figure with:
    - Bar chart (histogram) on primary y-axis
    - Line chart (P&L) on secondary y-axis

    Returns:
        Relative URL to plot file

    Note: Runs in thread pool to avoid blocking async event loop
    """
    from concurrent.futures import ThreadPoolExecutor
    import time

    timestamp = int(time.time())
    filename = f"{session_id}_{timestamp}.png"
    output_path = f"static/plots/{filename}"

    def _generate_plot():
        fig, ax1 = plt.subplots(figsize=(12, 6))

    # Histogram
    prices = [bin.midpoint for bin in price_distribution]
    counts = [bin.count for bin in price_distribution]
    ax1.bar(prices, counts, width=..., alpha=0.6, color='steelblue')
    ax1.set_ylabel('Frequency', color='steelblue')

    # P&L curve
    ax2 = ax1.twinx()
    payoffs = [strategy.total_payoff(p) for p in prices]
    ax2.plot(prices, payoffs, color='darkred', linewidth=3, label='P&L')
    ax2.axhline(0, color='black', linewidth=1, linestyle='--')
    ax2.set_ylabel('Profit/Loss ($)', color='darkred')

    # Current price marker
    ax2.axvline(current_price, color='green', linewidth=2,
                linestyle='--', label='Current Price')

        plt.title(f'{strategy.stock.symbol} Strategy Analysis')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Run in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _generate_plot)

    return f"/static/plots/{filename}"
```

## Frontend Architecture

### Modular JavaScript

**api.js** - API client
```javascript
export class AnalyzerAPI {
  constructor(baseUrl = '/api/v1') {
    this.baseUrl = baseUrl;
  }

  async loadStock(symbol) {
    const response = await fetch(`${this.baseUrl}/stock?symbol=${symbol}`);
    if (!response.ok) throw new Error(await response.text());
    return response.json();
  }

  async addOptionPosition(strike, right, quantity) {
    const response = await fetch(`${this.baseUrl}/strategy/option`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ strike, right, quantity })
    });
    return response.json();
  }

  async generatePlot(params) {
    const response = await fetch(`${this.baseUrl}/analyze/plot`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });
    return response.json();
  }
}
```

**ui.js** - DOM manipulation
```javascript
export function displayOptionChain(chainData) {
  const tbody = document.getElementById('chainTableBody');
  tbody.innerHTML = '';

  chainData.strikes.forEach(row => {
    const tr = createOptionRow(row);
    tbody.appendChild(tr);
  });
}

export function updateMetrics(metrics) {
  document.getElementById('expectedValue').textContent =
    `EV: $${metrics.expectedValue.toFixed(2)}`;
  document.getElementById('probProfit').textContent =
    `P(profit): ${(metrics.probabilityOfProfit * 100).toFixed(1)}%`;
}

export function showError(message) {
  const errorDiv = document.getElementById('errorMessage');
  errorDiv.textContent = message;
  errorDiv.style.display = 'block';
  setTimeout(() => errorDiv.style.display = 'none', 5000);
}
```

**state.js** - Frontend state
```javascript
class AppState {
  constructor() {
    this.sessionId = null;  // Set from server responses
    this.currentStock = null;
    this.currentChain = null;
    this.selectedExpiration = null;  // date string (YYYY-MM-DD)
    this.positions = [];
    this.parameters = {
      targetDate: null,  // Auto-set to earliest option expiration
      bins: 50,
      bootstrapSamples: null,  // null = direct, >0 = Monte Carlo
      historicalYears: 5,
      priceRangeMin: null,
      priceRangeMax: null,
      transactionCostPerContract: 0.65
    };
    this.loadingState = {
      stock: false,
      chain: false,
      plot: false,
      metrics: false
    };
    this.error = null;
  }

  setSession(sessionId) {
    this.sessionId = sessionId;
  }

  setExpiration(expirationDate) {
    this.selectedExpiration = expirationDate;
    this.parameters.targetDate = expirationDate;
  }

  addPosition(position) {
    this.positions.push(position);
  }

  updatePosition(index, position) {
    this.positions[index] = position;
  }

  removePosition(index) {
    this.positions.splice(index, 1);
  }

  clearPositions() {
    this.positions = [];
  }

  setLoading(key, isLoading) {
    this.loadingState[key] = isLoading;
  }

  setError(error) {
    this.error = error;
    setTimeout(() => this.error = null, 5000);
  }
}

export const appState = new AppState();
```

**main.js** - Application controller
```javascript
import { AnalyzerAPI } from './api.js';
import { displayOptionChain, updateMetrics, showError } from './ui.js';
import { appState } from './state.js';

const api = new AnalyzerAPI();

document.getElementById('loadStockBtn').addEventListener('click', async () => {
  const symbol = document.getElementById('symbolInput').value;
  try {
    const stock = await api.loadStock(symbol);
    appState.currentStock = stock;
    displayStockInfo(stock);
  } catch (error) {
    showError(error.message);
  }
});

// Event delegation for option position inputs
document.getElementById('chainTable').addEventListener('change', async (e) => {
  if (e.target.classList.contains('position-input')) {
    await handlePositionChange(e.target);
  }
});
```

## Configuration

**Environment Variables** (.env)
```bash
# IBKR API
IBKR_BASE_URL=https://localhost:5000/v1/api
IBKR_TIMEOUT=30

# Server
PORT=8080
HOST=0.0.0.0
RELOAD=true  # Development only

# Analysis Defaults
DEFAULT_BINS=50
DEFAULT_BOOTSTRAP=0
DEFAULT_BAR_SIZE=1w  # 1d, 1w, 1m
DEFAULT_HISTORICAL_YEARS=5
DEFAULT_TRANSACTION_COST=0.65  # Per contract

# Session Management
SESSION_TTL=3600  # 1 hour
SESSION_CLEANUP_INTERVAL=300  # 5 minutes

# Cache
CACHE_TYPE=memory  # memory | redis
REDIS_URL=redis://localhost:6379  # If using Redis

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json  # json | text
```

**config.py**
```python
from pydantic_settings import BaseSettings
from datetime import timedelta

class Settings(BaseSettings):
    ibkr_base_url: str
    ibkr_timeout: int = 30

    port: int = 8080
    host: str = "0.0.0.0"

    # Analysis defaults
    default_bins: int = 50
    default_bootstrap: int = 0
    default_bar_size: str = "1w"  # Historical data bar size
    default_historical_years: int = 5
    default_transaction_cost: float = 0.65  # Per contract

    # Session management
    session_ttl: int = 3600  # Session timeout in seconds (1 hour)
    session_cleanup_interval: int = 300  # Cleanup every 5 minutes

    cache_type: Literal["memory", "redis"] = "memory"
    redis_url: Optional[str] = None

    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "text"

    class Config:
        env_file = ".env"
```

## Testing Strategy

### Unit Tests (Target: >90% coverage)

**test_validation.py**
```python
def test_single_expiration_validation():
    """Ensure mixed expiration strategies are rejected"""
    call1 = OptionContract(strike=100, expiration=date(2024, 12, 20), ...)
    call2 = OptionContract(strike=105, expiration=date(2025, 1, 17), ...)

    strategy = Strategy(
        stock=stock,
        option_positions=[
            OptionPosition(contract=call1, quantity=1),
            OptionPosition(contract=call2, quantity=-1)
        ]
    )

    assert not strategy.validate_single_expiration()

def test_zero_quantity_rejected():
    """Zero quantity positions should be invalid"""
    with pytest.raises(ValidationError):
        OptionPosition(contract=call, quantity=0)

def test_missing_bid_ask_handling():
    """When bid/ask unavailable, use 0 (will trigger validation error)"""
    contract = OptionContract(strike=100, bid=None, ask=None, ...)
    pos = OptionPosition(contract=contract, quantity=1)
    # Should still calculate but premium_paid will be 0
    assert pos.premium_paid == 0
```

**test_statistics.py**
```python
def test_geometric_returns_basic():
    closes = np.array([100, 110, 121])
    returns = geometric_returns(closes, period=1)
    assert np.allclose(returns, [1.1, 1.1])

def test_bootstrap_deterministic():
    np.random.seed(42)
    returns = np.array([1.1, 0.9, 1.05])
    result = bootstrap_walk(returns, steps=3)
    assert isinstance(result, float)
    assert result > 0

def test_histogram_includes_max():
    values = np.array([1, 2, 3, 4, 5])
    bins = create_histogram(values, n_bins=3)
    # Verify max value (5) is in last bin
    assert bins[-1].upper > 5
    assert sum(b.count for b in bins) == len(values)

def test_direct_vs_bootstrap_returns():
    """Test that direct returns mode works differently from bootstrap"""
    closes = np.array([100, 110, 99, 105, 108])
    returns = geometric_returns(closes)

    # Direct: should return the actual returns array
    direct_prices = get_price_distribution(
        current_price=100,
        returns=returns,
        target_date=date.today() + timedelta(days=30),
        bootstrap_samples=None
    )
    assert len(direct_prices) == len(returns)

    # Bootstrap: should return N samples
    bootstrap_prices = get_price_distribution(
        current_price=100,
        returns=returns,
        target_date=date.today() + timedelta(days=30),
        bootstrap_samples=1000
    )
    assert len(bootstrap_prices) == 1000

def test_price_range_calculation():
    """Test auto price range with padding"""
    prices = np.array([90, 95, 100, 105, 110])
    min_p, max_p = calculate_price_range(prices)

    assert min_p == 90 * 0.95  # 5% below min
    assert max_p == 110 * 1.05  # 5% above max

    # Test user override
    min_p, max_p = calculate_price_range(prices, user_min=80, user_max=120)
    assert min_p == 80
    assert max_p == 120
```

**test_strategy.py**
```python
def test_long_call_payoff():
    call = OptionContract(
        strike=100,
        right="C",
        bid=5,
        ask=5.5,
        expiration=date(2024, 12, 20),
        multiplier=100
    )
    pos = OptionPosition(contract=call, quantity=1)

    # Below strike: lose premium (1 contract * $5.50 * 100 multiplier)
    assert pos.payoff_at_price(95) == -550

    # At strike: lose premium
    assert pos.payoff_at_price(100) == -550

    # Above strike: intrinsic - premium
    # Intrinsic: (110 - 100) * 100 = $1000
    # Premium paid: $550
    # Net: $450
    assert pos.payoff_at_price(110) == 450

def test_short_put_payoff():
    put = OptionContract(
        strike=100,
        right="P",
        bid=3,
        ask=3.5,
        expiration=date(2024, 12, 20),
        multiplier=100
    )
    pos = OptionPosition(contract=put, quantity=-1)  # Short

    # Above strike: keep premium (1 contract * $3 * 100)
    assert pos.payoff_at_price(105) == 300

    # Below strike: intrinsic loss + premium
    # Premium received: $300
    # Intrinsic loss: -(100 - 90) * 100 = -$1000
    # Net: $300 - $1000 = -$700
    assert pos.payoff_at_price(90) == -700

def test_covered_call():
    stock = Stock(
        symbol="AAPL",
        current_price=100,
        conid="265598",
        available_expirations=[date(2024, 12, 20)]
    )
    call = OptionContract(
        strike=105,
        right="C",
        bid=2,
        ask=2.5,
        expiration=date(2024, 12, 20),
        multiplier=100
    )

    strategy = Strategy(
        stock=stock,
        stock_quantity=100,
        option_positions=[
            OptionPosition(contract=call, quantity=-1)  # Short call
        ]
    )

    # At 110:
    # - Stock gains: 100 shares * ($110 - $100) = $1000
    # - Call intrinsic: -(110 - 105) * 100 = -$500
    # - Call premium received: $2 * 100 = $200
    # - Net: $1000 - $500 + $200 = $700
    assert strategy.total_payoff(110) == 700
```

**test_risk.py**
```python
def test_expected_value_weighted():
    bins = [
        Bin(lower=90, upper=95, count=10),   # 10% probability
        Bin(lower=95, upper=100, count=80),  # 80% probability
        Bin(lower=100, upper=105, count=10)  # 10% probability
    ]

    # Simple strategy: $1 gain at 100+, $0 otherwise
    strategy = create_mock_strategy(breakeven=100)

    ev = calculate_expected_value(strategy, bins)
    # Should be weighted toward middle bin
    assert ev > 0
```

**test_rate_limiter.py**
```python
@pytest.mark.asyncio
async def test_rate_limiter():
    """Ensure rate limiter prevents bursts"""
    limiter = RateLimiter(max_requests=5, per_seconds=1)

    start = time.time()
    for i in range(10):
        await limiter.acquire()

    elapsed = time.time() - start
    assert elapsed >= 1.0  # Should take at least 1 second for 10 requests
```

### Integration Tests

**test_ibkr_real.py** (optional, requires IBKR connection)
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_ibkr_connection():
    """Test against actual IBKR API (requires gateway running)"""
    client = IBKRClient(base_url="https://localhost:5000/v1/api", cache=MemoryCache())

    try:
        stock = await client.get_stock("AAPL")
        assert stock.symbol == "AAPL"
        assert stock.current_price > 0
    except IBKRAPIError as e:
        pytest.skip(f"IBKR not available: {e}")
```

**test_api_workflow.py**
```python
@pytest.mark.asyncio
async def test_full_strategy_workflow(test_client):
    # 1. Load stock
    response = await test_client.get("/api/v1/stock?symbol=AAPL")
    assert response.status_code == 200
    stock = response.json()

    # 2. Load option chain
    response = await test_client.get(f"/api/v1/chain?symbol=AAPL&month=DEC24")
    assert response.status_code == 200

    # 3. Add option position
    response = await test_client.post("/api/v1/strategy/option", json={
        "strike": 150,
        "right": "C",
        "quantity": -1
    })
    assert response.status_code == 200

    # 4. Generate plot
    response = await test_client.post("/api/v1/analyze/plot", json={
        "weeks": 12,
        "binCount": 50
    })
    assert response.status_code == 200

    # 5. Get metrics
    response = await test_client.get("/api/v1/analyze/metrics")
    assert response.status_code == 200
    metrics = response.json()
    assert "expectedValue" in metrics
    assert "probabilityOfProfit" in metrics

@pytest.mark.asyncio
async def test_mixed_expiration_rejected(test_client):
    """API should reject strategies with mixed expirations"""
    # Add first option
    await test_client.post("/api/v1/strategy/option", json={
        "strike": 150,
        "right": "C",
        "expiration": "2024-12-20",
        "quantity": 1
    })

    # Add second option with different expiration
    response = await test_client.post("/api/v1/strategy/option", json={
        "strike": 155,
        "right": "C",
        "expiration": "2025-01-17",
        "quantity": -1
    })

    # Should fail validation when trying to analyze
    response = await test_client.post("/api/v1/analyze/plot")
    assert response.status_code == 400
    data = response.json()
    assert data["code"] == "MIXED_EXPIRATION"
```

### Performance Tests

**test_benchmarks.py**
```python
import pytest
import time

def test_bootstrap_simulation_performance():
    """10,000 simulations should complete in <5 seconds"""
    returns = np.random.normal(1.01, 0.02, 1000)

    start = time.time()
    result = monte_carlo_simulation(
        returns,
        target_date=date.today() + timedelta(days=30),
        n_simulations=10000
    )
    elapsed = time.time() - start

    assert elapsed < 5.0
    assert len(result) == 10000

def test_large_option_chain_performance():
    """Processing 100 strikes should complete in <3 seconds"""
    # Test with mock data
    strikes = [OptionContract(...) for _ in range(100)]
    # ... measure processing time
```

### E2E Tests

**test_browser_flow.py** (Playwright)
```python
import pytest
from playwright.async_api import async_playwright

@pytest.mark.e2e
async def test_full_user_flow():
    """Test complete user workflow in browser"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # Load app
        await page.goto("http://localhost:8080")

        # Enter symbol
        await page.fill("#symbolInput", "AAPL")
        await page.click("#loadStockBtn")

        # Wait for loading state
        await page.wait_for_selector(".loading-spinner", state="hidden")

        # Verify stock loaded
        assert await page.text_content("#stockPrice")

        # Select expiration
        await page.select_option("#expirationSelect", "2024-12-20")

        # Add option position
        await page.fill("#strike150-call-quantity", "1")

        # Generate plot
        await page.click("#generatePlotBtn")
        await page.wait_for_selector("#plotImage")

        # Verify plot loaded
        assert await page.is_visible("#plotImage")

        await browser.close()
```

## Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Project setup (pyproject.toml, directory structure)
- [ ] Configuration management (including transaction costs, historical years)
- [ ] Domain models (with validation, single-expiration check)
- [ ] Custom exceptions (IBKRAPIError, validation errors)
- [ ] Rate limiter utility
- [ ] Unit tests for models and validation

### Phase 2: Statistical Engine (Week 1-2)
- [ ] Implement geometric returns, bootstrap
- [ ] Direct returns vs Monte Carlo modes
- [ ] Price range calculation (auto + user override)
- [ ] Histogram binning
- [ ] Risk metrics (EV, PoP) with transaction costs
- [ ] Comprehensive unit tests
- [ ] Performance benchmarks (bootstrap simulation)

### Phase 3: IBKR Integration (Week 2)
- [ ] HTTPX client with rate limiting
- [ ] Retry logic with exponential backoff
- [ ] Cache layer
- [ ] Stock/chain/history endpoints (configurable years)
- [ ] Error handling (IBKR down, symbol not found, rate limited)
- [ ] Mock-based integration tests
- [ ] Optional: Real IBKR integration tests

### Phase 4: API Layer (Week 2-3)
- [ ] Session state service with cookie management
- [ ] Session cleanup background task (every 5 min)
- [ ] Automatic target date selection (earliest expiration)
- [ ] Request validation (Pydantic schemas)
- [ ] Standardized error responses
- [ ] FastAPI routes with session dependency injection
- [ ] Position modification endpoints (PATCH, DELETE)
- [ ] Strategy summary endpoint
- [ ] Change reset to POST (not DELETE)
- [ ] API integration tests with validation checks

### Phase 5: Visualization (Week 3)
- [ ] Matplotlib chart generation (async with thread pool)
- [ ] Per-session plot file naming
- [ ] Plot directory creation and cleanup
- [ ] Plot caching
- [ ] Visual regression tests

### Phase 6: Frontend (Week 3-4)
- [ ] HTML/CSS structure
- [ ] Loading state UI (spinners, disabled buttons)
- [ ] Error display component
- [ ] Modular JavaScript with state management
- [ ] Position modification UI (edit, delete)
- [ ] Strategy summary display
- [ ] Analysis parameters controls (historical years, price range)
- [ ] Event handling
- [ ] Playwright E2E tests

### Phase 7: Polish (Week 4)
- [ ] Logging & observability
- [ ] Performance optimization
- [ ] Documentation
- [ ] Deployment guide

## Advanced Features (Future)

1. **Strategy Library**: Save/load named strategies (local storage or database)
2. **Mixed Expiration Support**: Multi-date analysis for calendar spreads
3. **Comparison View**: Side-by-side strategy analysis
4. **Greeks Calculator**: Delta, gamma, theta, vega
5. **Implied Volatility Surface**: IV skew modeling
6. **Early Assignment Risk**: Probability modeling for American options
7. **Real-time Updates**: WebSocket price feeds
8. **Portfolio View**: Multiple positions across symbols
9. **Backtesting**: Historical strategy performance
10. **Export**: PDF reports, CSV data
11. **Quick Strategy Presets**: One-click covered call, iron condor, etc.
12. **Position Exit Simulation**: Analyze selling before expiration
13. **Multi-user Support**: Redis session storage (if needed for deployment)

## Success Criteria

**Functional:**
- ✅ All API endpoints return proper HTTP status codes and standardized errors
- ✅ Single-expiration validation enforced
- ✅ Auto target date selection works correctly
- ✅ Bootstrap simulations complete in <5s for 10k samples
- ✅ Plots render correctly in browser with proper session isolation
- ✅ Expected value calculations are probability-weighted with transaction costs
- ✅ Premium calculations use correct bid/ask spread with validation
- ✅ Position modification (edit/delete) works seamlessly

**Performance:**
- ✅ Option chain loads in <3 seconds (with caching)
- ✅ Historical data cached effectively
- ✅ Rate limiting prevents IBKR API bans
- ✅ Plot generation doesn't block other requests

**Quality:**
- ✅ >90% unit test coverage
- ✅ All unit, integration, and E2E tests pass
- ✅ Performance benchmarks meet thresholds
- ✅ FastAPI auto-docs accessible at `/docs`

**UX:**
- ✅ Frontend shows loading states during API calls
- ✅ Errors displayed clearly with actionable messages
- ✅ Strategy summary shows capital requirements
- ✅ Mobile-responsive design

## Key Design Decisions

1. **NumPy over pure Python**: 10-100x faster for statistical operations
2. **Async throughout**: Better concurrency for IBKR API calls
3. **Expiration dates over weeks**: More precise time calculations, aligns with option reality
4. **Contract multiplier field**: Explicit modeling (default 100), extensible to non-standard contracts
5. **Cookie-based sessions**: Multi-tab support, simple migration path to Redis
6. **Auto target date selection**: UX improvement - defaults to earliest option expiration
7. **Single-expiration constraint**: Simplifies initial implementation, can extend to mixed later
8. **Bootstrap vs direct returns**: Flexibility in distribution modeling
9. **Configurable historical window**: Different market regimes have different characteristics
10. **Transaction costs included**: Realistic P&L modeling with $0.65/contract default
11. **Price range auto-calculation**: Bootstrap min/max with 5% padding, user-overridable
12. **Per-session plot files**: Prevents race conditions in multi-tab scenarios
13. **Rate limiting with retry**: Protects against IBKR API bans
14. **Matplotlib in thread pool**: Prevents blocking async event loop
15. **Probability-weighted EV**: More accurate than uniform bin averaging
16. **Bid/ask spread modeling**: Realistic transaction costs, handles missing prices with validation
17. **Protocol-based cache**: Easy to swap memory → Redis
18. **Modular frontend**: No framework lock-in
19. **Single-user local deployment**: No auth complexity, suitable for personal use

## Deployment

**Development**:
```bash
# Ensure static/plots directory exists
mkdir -p static/plots

uvicorn src.option_analyzer.main:app --reload --port 8080
```

**Production** (single-user local deployment):
```bash
# Single worker only (in-memory sessions)
uvicorn src.option_analyzer.main:app \
  --host 127.0.0.1 \
  --port 8080 \
  --workers 1 \
  --log-config logging.json
```

**Note**: This application is designed for **single-user local deployment only**:
- No authentication/authorization
- In-memory session storage (not Redis)
- Single worker process (--workers 1)
- Bind to localhost (127.0.0.1) for security
- Not suitable for public internet deployment without additional security

**Docker**:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
CMD ["uvicorn", "src.option_analyzer.main:app", "--host", "0.0.0.0"]
```
