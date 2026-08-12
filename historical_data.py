"""
HISTORICAL DATA PIPELINE
===========================
Pulls driver (overnight) history and ASX outcome history from yfinance,
and computes the Step-2 outcome fields. Nulls any return calculated
across an abnormal gap in the raw source data rather than trusting it
(catches things like the S&P500 12-day hole and Brent 17-day hole found
during validation, and any future undiscovered ones, automatically).

10:15 / 10:30 outcome columns exist in the schema now and are populated
NaN — Phase 2 fills them in from live daily snapshots once that starts
running, no restructure needed later.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import time

from config_v1 import DRIVERS, PER_OBSERVATION_GAP_NULL_THRESHOLD_DAYS


def fetch_raw_history(ticker, max_retries=3):
    """Pulls full daily history for a ticker, with retry/backoff on
    rate-limit errors (never treat a Yahoo throttle as 'no data')."""
    for attempt in range(max_retries):
        try:
            hist = yf.Ticker(ticker).history(period="max")
            return hist
        except Exception as e:
            if "Too Many Requests" in str(e) or "rate limit" in str(e).lower():
                time.sleep(3 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"{ticker}: exhausted retries on rate limiting")


def compute_valid_pct_change(hist, gap_threshold_days=PER_OBSERVATION_GAP_NULL_THRESHOLD_DAYS):
    """Returns a Series of % change in Close, indexed by date, with any
    change computed across a gap wider than gap_threshold_days set to
    NaN instead of a fabricated large move."""
    if hist.empty:
        return pd.Series(dtype=float)
    hist = hist.dropna(subset=["Close"]).copy()
    dates = pd.to_datetime(hist.index.date)
    pct_change = hist["Close"].pct_change().to_numpy() * 100
    gap_days = pd.Series(dates).diff().dt.days.to_numpy()
    invalid = gap_days > gap_threshold_days
    pct_change[invalid] = np.nan
    return pd.Series(pct_change, index=dates, name="pct_change")


def build_driver_table(driver_names=None):
    """Wide table: index = driver's own trading date, one column per
    named driver, values = % change (NaN where nulled or missing).
    NOT yet aligned to ASX sessions — see align_to_asx_sessions."""
    driver_names = driver_names or list(DRIVERS.keys())
    series_list = []
    for name in driver_names:
        ticker, role, first_available, notes = DRIVERS[name]
        hist = fetch_raw_history(ticker)
        s = compute_valid_pct_change(hist)
        s.name = name
        series_list.append(s)
    table = pd.concat(series_list, axis=1)
    table.index.name = "date"
    return table.sort_index()


def align_to_asx_sessions(driver_table, asx_dates):
    """For each ASX trading date, attach the most recent PRIOR driver
    row (strictly before that date). This is a date-only backward join
    — since Yahoo's daily bars are already indexed by each exchange's
    own local trading date, taking 'most recent prior date' correctly
    handles weekends, holidays, and the AU/US date-line offset without
    needing explicit timezone or DST math at daily granularity. (Note:
    Phase 2's intraday 10:15/10:30 alignment DOES need explicit
    timezone/DST handling, since that operates on wall-clock cutoffs,
    not whole trading dates — this simplification applies to Phase 1 only.)
    """
    asx_dates = pd.to_datetime(sorted(asx_dates))
    driver_table = driver_table.sort_index()

    left = pd.DataFrame({"asx_date": asx_dates})
    right = driver_table.reset_index()
    right = right.rename(columns={right.columns[0]: "driver_date"})  # positional, not name-dependent

    merged = pd.merge_asof(
        left, right,
        left_on="asx_date", right_on="driver_date",
        direction="backward", allow_exact_matches=False,
    )
    merged = merged.set_index("asx_date").drop(columns=["driver_date"])
    return merged


def compute_asx_outcomes(ticker, gap_threshold_days=PER_OBSERVATION_GAP_NULL_THRESHOLD_DAYS):
    """Per-ASX-ticker outcome table. Columns match the Step 2 spec.
    10:15/10:30-dependent columns are present but NaN until Phase 2."""
    hist = fetch_raw_history(ticker)
    if hist.empty:
        return pd.DataFrame()
    hist = hist.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    dates = pd.to_datetime(hist.index.date)
    hist.index = dates

    gap_days = pd.Series(dates).diff().dt.days.to_numpy()
    row_valid = gap_days <= gap_threshold_days  # False for row 0 (no prior) and abnormal gaps

    prev_close = hist["Close"].shift(1)
    open_ = hist["Open"]
    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]

    out = pd.DataFrame(index=hist.index)
    out["prev_close"] = prev_close
    out["open"] = open_
    out["close"] = close
    out["high"] = high
    out["low"] = low

    gap_pct = (open_ - prev_close) / prev_close * 100
    gap_pct[~row_valid] = np.nan
    out["gap_pct"] = gap_pct  # prev_close -> open

    out["open_to_close_pct"] = (close - open_) / open_ * 100
    out["mfe_pct"] = (high - open_) / open_ * 100   # raw, LONG-perspective; sign-flipped for SHORT at analysis time
    out["mae_pct"] = (low - open_) / open_ * 100

    # next-session / 2-day / 3-day returns, measured close-to-close,
    # stored on the row where the setup would have been identified
    close_fwd1 = close.shift(-1)
    close_fwd2 = close.shift(-2)
    close_fwd3 = close.shift(-3)
    out["next_session_return"] = (close_fwd1 - close) / close * 100
    out["day2_return"] = (close_fwd2 - close) / close * 100
    out["day3_return"] = (close_fwd3 - close) / close * 100

    # Phase 2 placeholders — schema exists now, filled later, no
    # restructure needed when live intraday capture starts.
    out["price_10_15"] = np.nan
    out["price_10_30"] = np.nan
    out["prevclose_to_1015_pct"] = np.nan
    out["prevclose_to_1030_pct"] = np.nan
    out["open_to_1015_pct"] = np.nan
    out["open_to_1030_pct"] = np.nan
    out["t1015_to_close_pct"] = np.nan
    out["t1030_to_close_pct"] = np.nan

    return out


def build_asx_outcome_tables(tickers):
    """Dict of ticker -> outcome DataFrame, for a list of ASX tickers."""
    return {t: compute_asx_outcomes(t) for t in tickers}
