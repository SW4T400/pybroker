# Yield Accrual & Funding Rate Payments

PyBroker now supports simulating **staking/earn yield** (coin-denominated) and
**funding rate payments** (cash-denominated) for crypto carry trade backtesting.

---

## Quick Start

```python
import pybroker as pyb
from decimal import Decimal

def after_exec(ctxs):
    # --- Funding rate on short perp (cash payment) ---
    ctx = ctxs.get('BTC_PERP')
    if ctx is not None:
        pos = ctx.short_pos()
        if pos is not None and pos.shares > 0:
            rate = Decimal(str(ctx.indicator('funding_rate')[-1]))
            notional = pos.shares * Decimal(str(ctx.close[-1]))
            payment = notional * rate  # positive rate = shorts receive
            ctx.add_cash_flow(float(payment))  # default: 8dp truncation

    # --- Earn yield on spot (share accrual) ---
    ctx = ctxs.get('BTC')
    if ctx is not None:
        pos = ctx.long_pos()
        if pos is not None and pos.shares > 0:
            daily_rate = 0.000014  # ~0.5% APY / 365
            yield_shares = float(pos.shares) * daily_rate
            ctx.accrue_yield(yield_shares)  # default: 8dp truncation

strategy = pyb.Strategy(data_source, start_date=..., end_date=...)
strategy.add_execution(exec_fn, ['BTC', 'BTC_PERP'])
strategy.set_after_exec(after_exec)
result = strategy.backtest(warmup=20)
```

---

## API Reference

### `ctx.accrue_yield(shares, symbol=None, precision='0.00000001')`

Accrues yield shares on an existing **long** position.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shares` | `int/float/Decimal` | — | Number of shares to accrue |
| `symbol` | `str` or `None` | `None` | Ticker symbol (defaults to context's symbol) |
| `precision` | `str/Decimal` or `None` | `'0.00000001'` (8dp) | Step size for rounding down. Pass `None` to disable rounding. |

**How it works:**
- Creates a zero-cost-basis `Entry` (price=0) appended to the position's entries deque
- When sold via FIFO, these shares generate full proceeds as profit
- Consecutive daily accruals consolidate into a single yield entry per symbol
- Returns the yield `Entry` if successful, `None` if position doesn't exist or shares round to zero

**PnL behavior:**
- Equity curve: **correct** (pos.shares * close includes accrued shares)
- Trade records: **correct** (yield shares have entry price=0, so PnL = shares * exit_price)
- Portfolio metrics (Sharpe, drawdown): **correct** (derived from equity curve)

### `ctx.add_cash_flow(amount, symbol=None, precision='0.00000001')`

Adds a cash flow to the portfolio.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `amount` | `int/float/Decimal` | — | Cash amount (positive=inflow, negative=outflow) |
| `symbol` | `str` or `None` | `None` | Ticker symbol for logging |
| `precision` | `str/Decimal` or `None` | `'0.00000001'` (8dp) | Step size for rounding down. Pass `None` to disable rounding. |

Returns the new cash balance.

### `portfolio.yield_records`

A `deque[YieldRecord]` containing all yield and funding records:

```python
# After backtest
import pandas as pd
yield_df = pd.DataFrame(result.portfolio.yield_records)
# Columns: date, symbol, type ('yield'|'funding'), shares, amount
```

---

## Precision / Truncation

Both `accrue_yield` and `add_cash_flow` default to **8 decimal place truncation**
(`precision='0.00000001'`), matching Binance's universal rounding rule for both
funding payments and earn yields. You generally don't need to override this.

The `precision` parameter rounds amounts **down** (never up) using `ROUND_DOWN`:

```python
# Default behavior (8dp) — no precision argument needed:
ctx.accrue_yield(0.000020549999)   # → 0.00002054
ctx.add_cash_flow(12.345678901)    # → 12.34567890

# Override for special cases:
ctx.accrue_yield(547.945, precision='1')      # SHIB: → 547 whole coins
ctx.add_cash_flow(amount, precision=None)      # No rounding at all
```

If truncation reduces the amount to zero, no accrual happens and `None` is returned.

---

## Execution Timing

The `after_exec` callback runs **after** `capture_bar` and your trading logic,
but **before** orders are scheduled:

```
Per-bar execution order:
1. check_stops          -- stops see previous bar's accrued shares
2. place orders         -- fills from previous bar's signals
3. capture_bar          -- records equity (shares * close)
4. before_exec
5. exec_fns             -- your trading signals
6. after_exec           -- ACCRUE YIELD / FUNDING HERE
7. schedule orders
8. incr_bars
```

Yield accrued in step 6 is reflected in the **next** bar's `capture_bar` (step 3).
This one-bar lag is correct for daily accrual.

**Sell timing:** If you signal a sell on day X (step 5), the sell executes on
day X+1 (step 2, via `sell_delay=1`). Your `after_exec` on day X still runs
(position exists), so yield IS earned on day X. On day X+1, the sell fills
before `after_exec`, so no yield on the execution day.

---

## Binance Mechanics Reference

### Funding Rate (USDS-M / USDC-M Perpetual Futures)

**Rate formula:**
```
Funding Rate = Premium Index + clamp(Interest Rate - Premium Index, -cap, +cap)
```

- **Interest Rate:** Fixed at 0.0001 (0.01%) per 8-hour interval across all symbols
- **Premium Index:** TWAP of `(Impact Mid Price - Index Price) / Index Price`
  over the funding interval

**Payment formula:**
```
Funding Payment = Mark Price * Position Size * Funding Rate
```

**Settlement frequency** (per-symbol, confirmed from live API):

| Interval | Settlement Times (UTC) | Symbols |
|----------|----------------------|---------|
| 8-hour | 00:00, 08:00, 16:00 | ~180 (majors: BTCUSDT, ETHUSDT) |
| 4-hour | 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 | ~520 (most altcoins) |
| 1-hour | Every hour | ~2 (newly listed/volatile) |

**Rate caps** (per-symbol, symmetric):

| Cap | Example Symbols |
|-----|----------------|
| +/-0.3% | BTCUSDT, ETHUSDT |
| +/-0.375% | SOLUSDT, XRPUSDT |
| +/-0.75% | AAVEUSDT, SUIUSDT |
| +/-2% | Most small-cap alts (default) |
| +/-3% | BTCDOMUSDT, special products |

**Direction:** Positive rate = longs pay shorts; negative = shorts pay longs

**Precision:** All rates and payments use 8 decimal places. Funding payment
amounts are truncated to 8dp.

**For daily bars:** Sum all funding payments for the day. For 8-hour symbols,
that's 3 payments; for 4-hour symbols, 6 payments.

### Simple Earn (Flexible Products)

**Daily yield formula:**
```
Daily Interest = Balance * APR / 365
```

**Per-minute formula** (for real-time accrual):
```
Per-Minute Interest = Balance * APR / (365 * 24 * 60)
```

- **Rates are APR** (simple, not compounded). The displayed
  `latestAnnualPercentageRate` is APR. With auto-subscribe enabled,
  effective yield compounds: `APY = (1 + APR/365)^365 - 1`
- **Tiered rates:** Higher balances get lower rates
  (e.g., `0-5 BTC: 5%`, `5-10 BTC: 3%`)
- **Rounding:** Truncated (rounded down) to 8 decimal places
- **Distribution:** Accrued per minute (real-time rewards). Daily totals
  credited to Earn Wallet, typically between 00:00-08:00 UTC
- **T+1 model:** Interest begins day after subscription. No interest on
  redemption day
- **Asset:** Paid in the staked coin (BTC earns BTC, ETH earns ETH)
- **No lock-up** for Flexible Products; instant subscribe/redeem

**Reward types tracked by Binance:**
1. `realTimeRewards` -- base APR interest
2. `bonusRewards` -- additional bonus
3. `airdropRewards` -- airdrops (may be in different asset, e.g., BETH for ETH)

---

## Shorts & Funding Rate Direction

`add_cash_flow` is **position-agnostic** — it modifies `Portfolio.cash` directly regardless
of whether you hold a long, short, or no position at all. The sign of the amount controls
the direction:

- **Positive amount** → cash inflow (you receive money)
- **Negative amount** → cash outflow (you pay money)

For funding rates, the direction is determined by the rate sign:

| Rate Sign | Longs | Shorts |
|-----------|-------|--------|
| Positive (`+0.01%`) | Pay funding | **Receive** funding |
| Negative (`-0.01%`) | **Receive** funding | Pay funding |

In a typical carry trade (short perp + long spot), positive funding rates generate
cash income on the short leg:

```python
# payment = notional * rate
# If rate > 0 and you're short: payment > 0 → cash inflow
# If rate < 0 and you're short: payment < 0 → cash outflow
ctx.add_cash_flow(float(payment))
```

**Negative cash balance:** If funding payments drain your cash below zero, pybroker
does **not** auto-liquidate or exit positions. Cash can go negative and the backtest
continues. If you want to simulate margin calls or forced exits, add that logic
yourself in `after_exec`.

---

## Feeding Data Into the Backtest

Funding rates and earn yields are **not applied automatically**. You provide the data
and write the `after_exec` callback logic. Two approaches:

### Option 1: Custom Columns (recommended for daily bars)

```python
import pybroker as pyb

# Register custom columns so they appear as ctx attributes
pyb.register_columns('funding_rate', 'earn_apr')

# Your DataFrame must include these columns alongside OHLCV:
# df['funding_rate'] = daily sum of all settlement periods
# df['earn_apr'] = daily APR (e.g. 0.05 for 5%)

def after_exec(ctxs):
    # Funding on short perp
    ctx = ctxs.get('BTC_PERP')
    if ctx is not None:
        pos = ctx.short_pos()
        if pos is not None and pos.shares > 0:
            rate = ctx.funding_rate[-1]       # custom column as attribute
            payment = float(pos.shares) * ctx.close[-1] * rate
            ctx.add_cash_flow(payment)

    # Earn yield on spot
    ctx = ctxs.get('BTC')
    if ctx is not None:
        pos = ctx.long_pos()
        if pos is not None and pos.shares > 0:
            apr = ctx.earn_apr[-1]            # custom column as attribute
            yield_shares = float(pos.shares) * apr / 365
            ctx.accrue_yield(yield_shares)
```

### Option 2: Indicators

```python
funding_ind = pyb.indicator('funding_rate', lambda data: data['funding_rate'])
# Then in after_exec: rate = ctx.indicator('funding_rate')[-1]
```

Option 1 is simpler — `register_columns` makes any extra DataFrame column accessible
as `ctx.<column_name>` in your callbacks.

---

## Data Preparation

### Funding Rate Data

```python
import pybroker as pyb

pyb.register_columns('funding_rate')

# Your DataFrame must include a 'funding_rate' column
# For daily bars with 8-hour settlement, sum 3 rates:
df['funding_rate'] = df['fr_00'] + df['fr_08'] + df['fr_16']
# For 4-hour settlement symbols, sum 6 rates:
df['funding_rate'] = df[['fr_00','fr_04','fr_08','fr_12','fr_16','fr_20']].sum(axis=1)
```

Historical funding data sources:
- Binance API: `GET /fapi/v1/fundingRate` (up to 1000 records per request)
- Binance API: `GET /fapi/v1/fundingInfo` (settlement frequency and caps per symbol)
- CoinGlass API (aggregated across exchanges)

### Earn Yield Data

Earn yield rates are not available as structured historical data. Options:
- Use DeFi lending rates (Aave, Compound) as proxy
- Use a conservative fixed assumption (e.g., 0.5% APY for BTC)
- Scrape/collect rates over time from Binance UI
- Query `GET /sapi/v1/simple-earn/flexible/list` for current rates and tier structure

---

## Accessing Yield Records After Backtest

```python
import pandas as pd

result = strategy.backtest(warmup=20)

# Access yield records from the after_exec function's closure
# or from the portfolio directly (if you kept a reference)
yield_df = pd.DataFrame(after_exec.yield_log)  # if using the log pattern

# Per-symbol totals
print(yield_df.groupby(['symbol', 'type']).agg({
    'shares': 'sum',
    'amount': 'sum',
}))
```

---

## Internal Design

### Zero-Cost-Basis Approach

Yield shares are added as `Entry(price=0, is_yield=True)`. This means:

- `_exit_long` computes: `pnl = shares * fill_price - shares * 0 = full proceeds`
- No modification needed to existing PnL, stop, or FIFO sell logic
- Division-by-zero guard added for `return_pct` when `entry.price == 0`

### Invariant

The critical invariant `pos.shares == sum(entry.shares for entry in pos.entries)`
is maintained by:
- Adding yield to both `pos.shares` and the yield entry's `entry.shares`
- Cleanup of `_yield_entries` dict when entries are consumed by FIFO sells,
  stop-triggered exits, or full position closure

### Entry Consolidation

Instead of creating one entry per accrual day (which would bloat the deque),
consecutive yield accruals on the same symbol consolidate into a single yield
entry. If that entry is consumed by a sell, the next accrual creates a fresh one.

---

## Files Modified

| File | Changes |
|------|---------|
| `src/pybroker/portfolio.py` | `Entry.is_yield`, `YieldRecord`, `Portfolio.accrue_yield()`, `Portfolio.add_cash_flow()`, `precision` parameter, div-by-zero guard |
| `src/pybroker/context.py` | `ExecContext.accrue_yield()`, `ExecContext.add_cash_flow()` |
| `src/pybroker/__init__.py` | Export `YieldRecord` |
| `tests/test_portfolio.py` | 22 new tests (basic, FIFO, PnL, precision, cumulative, default 8dp) |
