# PyBroker Deep Audit: Changes, Reasoning & Dropped Cases

**Date:** 2026-03-08
**Scope:** Full codebase audit for bugs, best-practice violations, and performance

---

## Methodology

Four parallel deep-dive agents examined:
1. **Core architecture** (strategy.py, portfolio.py, context.py, scope.py)
2. **Data pipeline & performance** (indicator.py, vect.py, data.py, eval.py, cache.py)
3. **Order execution & edge cases** (portfolio.py order flow, slippage.py, common.py)
4. **Tests & documentation** (test coverage gaps, doc discrepancies)

Every finding was then **re-verified** by a second round of agents that checked whether the proposed fix was safe and wouldn't destroy existing behavior. Only changes that survived this scrutiny were applied.

---

## Applied Changes

### 1. BUG FIX: Walkforward Session State Leakage

**File:** `src/pybroker/strategy.py` (method `_run_walkforward`)

**What was wrong:**
The `sessions` dictionary (a `defaultdict(dict)`) was created *once* before the walkforward loop and passed to `backtest_executions` on every window iteration. This meant any user state stored via `ctx.session` in window 1 would persist into window 2, violating walkforward independence.

**Why this is a bug (not a feature):**
- The `portfolio` object IS intentionally reused across windows (cumulative P&L is correct behavior for walkforward)
- But `sessions` serves a different purpose: it stores per-bar custom user state
- Docs say sessions "persist for each bar during the execution" -- implying per-window scope
- Models are retrained per window, so session data from prior windows is stale
- No tests verify cross-window session persistence
- No documentation describes this as intended behavior

**What was changed:**
- Moved `sessions = defaultdict(dict)` from before the loop to inside the loop body, so each walkforward window gets fresh session state

**Risk assessment:**
Only users who explicitly relied on undocumented session leakage across windows would be affected. This is extremely unlikely given the lack of any documentation or tests for this behavior.

---

### 2. PERFORMANCE: Batch pd.concat in Walkforward

**File:** `src/pybroker/strategy.py` (method `_run_walkforward`)

**What was wrong:**
Inside the walkforward loop, `pd.concat([signals[sym], signals_df])` was called on every iteration, creating intermediate DataFrame copies that grow with each window.

**What was changed:**
- Collect DataFrames in a `defaultdict(list)` during the loop
- Single `pd.concat(dfs)` per symbol after the loop completes

**Why this is safe:**
- `signals[sym]` was never read during the loop (only written to)
- The final output is identical -- same DataFrames in the same order
- `pd.concat` on a list of N items is O(n) total; repeated binary concat is O(n^2)

---

### 3. CODE QUALITY: Float-to-Decimal in RandomSlippageModel

**File:** `src/pybroker/slippage.py`

**What was wrong:**
`Decimal(random.uniform(self.min_pct, self.max_pct))` converts a float to Decimal by capturing its full IEEE 754 binary representation. For example, `Decimal(0.1)` becomes `Decimal('0.1000000000000000055511151231257827...')` instead of `Decimal('0.1')`.

**Why this is NOT a functional bug (but still worth fixing):**
- In 90%+ of cases, `int()` truncation in `_get_shares()` eliminates the precision garbage
- Exchange precision rounding handles remaining cases
- Even when preserved, the financial impact is ~$0.00000001 on $100 orders

**Why we fixed it anyway:**
- Violates the codebase's own pattern (`to_decimal()` uses `Decimal(str(...))`)
- `Decimal(float)` is explicitly warned against in Python docs
- The existing test mocks `random.uniform` to return a string, so it never catches this

**What was changed:**
`Decimal(random.uniform(...))` -> `Decimal(str(random.uniform(...)))`

---

### 4. PERFORMANCE: Pre-group DataFrame by Symbol in Indicator Computation

**File:** `src/pybroker/indicator.py` (methods `compute_indicators` and `__call__`)

**What was wrong:**
Two hot paths filtered the full DataFrame per symbol using `df[df[col] == sym]`, resulting in O(n*k) boolean scans (n=rows, k=symbols). For 100 symbols on a 1M-row DataFrame, that's 100 full scans.

**What was changed:**
- Create a single `grouped = df.groupby(DataCol.SYMBOL.value)` before the loop
- Replace `df[df[col] == sym]` with `grouped.get_group(sym)`

**Why this is safe:**
- DataFrame has RangeIndex at these points (`reset_index()` is always called upstream)
- All accessed symbols are guaranteed to exist in the DataFrame
- Row order within groups is preserved by pandas groupby
- Output dtypes are identical
- Edge case: if a symbol doesn't exist, `get_group()` raises KeyError -- but the code only iterates over symbols known to exist in the DataFrame

---

### 5. PERFORMANCE: Remove Unnecessary `copy=True` from `to_numpy()`

**Files:** `src/pybroker/indicator.py` and `src/pybroker/eval.py`

**Git history context:** Commit `b615898` ("Fix readonly NumPy arrays from Pandas") added `copy=True` as a blanket fix across 5 files to prevent crashes from Pandas 2.0+ returning read-only views. The fix was applied uniformly to ALL `.to_numpy()` calls, even where downstream code never writes to the arrays. Some of these copies also serve as a defragmentation benefit (contiguous, independently-allocated arrays are better for Numba and CPU cache). We only removed copies where both conditions hold: (1) arrays are never written to, AND (2) the arrays already come from computed/filtered results that are contiguous and independent of the source DataFrame.

**What was wrong:**
`to_numpy(copy=True)` forces a deep copy of the underlying array. In several locations, these arrays are only read, never modified -- making the copy pure waste.

**Where copies were removed (safe):**
- `indicator.py:226` -- arrays stored in a temporary local dict, passed read-only to Numba `@njit` functions that always create new output arrays via `np.zeros()`
- `eval.py:865-883, 961` -- arrays passed to read-only `@njit` evaluation functions (sharpe_ratio, max_drawdown, etc.) and bootstrap sampling that creates its own working copies

**Where copies were KEPT (essential -- do NOT remove):**
- `scope.py:285-286` (`ColumnScope.fetch_dict`) -- arrays are cached in `_sym_cols` and returned as slices to user code. If users modify the slice, they'd corrupt the cache. The copy is a defensive measure.
- `scope.py:375-377` (`IndicatorScope.fetch`) -- same pattern: arrays cached in `_sym_inds`, returned as slices to user callbacks.

---

### 6. PERFORMANCE: Vectorized Indicator Result Aggregation

**File:** `src/pybroker/indicator.py` (method `__call__`)

**What was wrong:**
DataFrame construction used Python-level `list.extend()` with `itertools.repeat()`, building columns element-by-element before converting to DataFrame. This is slow for large datasets due to Python object overhead.

**What was changed:**
- Use `np.repeat(sym_list, counts)` for the symbol column
- Use `np.concatenate(date_arrays)` for dates
- Use `np.concatenate(arrays)` for each indicator
- Construct DataFrame from numpy arrays directly

**Why this is safe:**
- Output dtypes are preserved (datetime64, float64, etc.)
- Row order is identical (same symbol iteration order)
- The `from_dict()` constructor handles numpy arrays the same as lists

---

## Dropped Cases (investigated but NOT changed)

### A. Division-by-Zero Guard in `_exit_short` (portfolio.py:726)

**Original claim:** `return_pct = ((entry.price / fill_price) - 1) * 100` has no guard for `fill_price == 0`, unlike `_exit_long` which guards `entry.price == 0`.

**Why dropped:** The asymmetry is intentional and correct:
- `fill_price` is validated `> 0` upstream in `_verify_input()` (line 484-485)
- `entry.price` for shorts is always `> 0` because short entries are created with `price=fill_price` (already validated)
- The `_exit_long` guard exists specifically for **yield entries** which have `price=0` -- but yield entries are only created for longs (`type='long'` in `_create_yield_entry`), never shorts
- Adding an unnecessary guard would be misleading (suggests shorts can have zero-price entries when they can't)

### B. `add_cash_flow` Recording Zero-Amount YieldRecords (portfolio.py:1226)

**Original claim:** After precision truncation, amount can become 0, but unlike `accrue_yield`, `add_cash_flow` still records it.

**Why dropped:** Recording zero amounts is valid audit trail behavior:
- `self.cash += 0` is a no-op
- Zero-amount `YieldRecord` entries don't affect any downstream calculations
- Maintaining a complete audit trail (including "attempted but rounded to zero" entries) is a reasonable design choice
- Existing tests explicitly verify this behavior passes

### C. O(n) `list.remove()` in `check_stops` (portfolio.py:1311)

**Original claim:** `pos.entries.remove(entry)` is O(n) per removal.

**Why dropped:**
- `entries` is a `deque` with FIFO semantics critical to position management
- Positions rarely have >10 entries in practice
- Switching data structures would risk breaking FIFO exit order
- The theoretical O(n) cost is negligible at real-world scale

### D. `.isin()` Replacement with `searchsorted` (scope.py:376)

**Original claim:** `.isin(filter_dates)` is O(n*m); `searchsorted` would be faster.

**Why dropped:**
- No guarantee that the indicator series index is sorted or is a DatetimeIndex
- `.isin()` works correctly regardless of index order, type, or duplicates
- `searchsorted` requires sorted input -- violating this silently produces wrong results
- The safety risk outweighs the performance gain

### E. `copy=True` in `scope.py` (lines 285-286 and 375-377)

**Original claim:** These copies are unnecessary.

**Why dropped:** They are **essential defensive measures**:
- Both `ColumnScope` and `IndicatorScope` cache arrays and return slices to user code
- If users modify returned slices (e.g., `data[:] = 0`), the mutation propagates to the cache
- Subsequent calls would get corrupted cached data
- The copy breaks the reference chain between cache and user-visible data

### F. Pre-slice `reset_index()` in `ColumnScope` (scope.py:279)

**Original claim:** `reset_index()` called on every symbol access; should pre-slice.

**Why dropped:**
- The lazy loading pattern is intentional for large datasets
- Not all symbols may be queried during a backtest
- Pre-slicing in `__init__` would front-load memory allocation for unused symbols
- The current approach is a space-time tradeoff favoring memory efficiency

### G. Short Return % Formula "Inconsistency" (portfolio.py:726 vs 905)

**Original claim:** Long uses `(fill/entry - 1)`, short uses `(entry/fill - 1)` -- "inverted".

**Why dropped:** Both formulas are mathematically correct for their respective position types:
- Long return: `(sell_price / buy_price) - 1` = price appreciation
- Short return: `(entry_price / cover_price) - 1` = inverse price movement profit
- Example: Short at 100, cover at 90 -> `(100/90 - 1) * 100 = 11.11%` profit (correct)
- Example: Short at 100, cover at 110 -> `(100/110 - 1) * 100 = -9.09%` loss (correct)
- This is standard practice in quantitative finance

---

## Test Results

All **4089 tests pass** after changes (0 failures, 0 errors).

## Files Modified

| File | Changes |
|------|---------|
| `src/pybroker/strategy.py` | Session isolation fix, batched concat |
| `src/pybroker/slippage.py` | Decimal conversion fix |
| `src/pybroker/indicator.py` | groupby optimization, copy removal, vectorized aggregation |
| `src/pybroker/eval.py` | copy removal |
