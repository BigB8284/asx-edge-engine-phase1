"""
INTRADAY DATA — paginated fetch + quality filtering
========================================================
Fetches full 5-minute intraday history from EODHD (paginated, since
the API caps 5m requests at 600 days per call) and applies the SAME
quality checks proven in eodhd_intraday_validation.py — Sydney-timezone
DST-aware classification, continuous-session isolation, incomplete-day
exclusion, and implausible-move flagging. Nothing here re-implements
that logic; it's imported directly.

Output: for each ticker, a dict of {date: bars_df} for CLEAN days only
(complete, continuous-session bars, minutes_from_open computed) plus
lists of excluded days and flagged moves — visible, not silently
dropped, per the standing project rule.
"""

import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

from eodhd_logic import (
    to_sydney_and_classify, day_completeness_report, outside_session_summary,
    flag_implausible_moves,
)

BASE_URL = "https://eodhd.com/api/intraday/{ticker}"
MAX_DAYS_PER_CALL = 600  # EODHD's documented cap for 5-minute bars


def _fetch_one_window(ticker, api_token, from_dt, to_dt, max_retries=3):
    """One paginated request, with retry/backoff on rate limiting —
    same resilience pattern as historical_data.fetch_raw_history:
    never crash the whole pull on one bad window, return (None, reason)
    instead so a failure is diagnosable, not just a silent gap."""
    params = {"api_token": api_token, "interval": "5m", "fmt": "json",
              "from": int(from_dt.timestamp()), "to": int(to_dt.timestamp())}
    last_reason = "unknown"
    for attempt in range(max_retries):
        try:
            resp = requests.get(BASE_URL.format(ticker=ticker), params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return (data if data else [], None)
            if "Too Many Requests" in resp.text or resp.status_code == 429:
                last_reason = f"HTTP 429 rate limited: {resp.text[:200]}"
                time.sleep(3 * (attempt + 1))
                continue
            return (None, f"HTTP {resp.status_code}: {resp.text[:300]}")
        except Exception as e:
            last_reason = f"Exception: {e}"
            time.sleep(2)
            continue
    return (None, last_reason)


def fetch_full_intraday_history(ticker, api_token, start_date, end_date):
    """Paginates across MAX_DAYS_PER_CALL windows to cover the full
    requested range. Returns (all_bars_list, failed_windows) — failed
    windows are reported WITH THE ACTUAL REASON, not silently skipped."""
    all_bars = []
    failed_windows = []
    window_start = start_date
    while window_start < end_date:
        window_end = min(window_start + timedelta(days=MAX_DAYS_PER_CALL - 1), end_date)
        data, reason = _fetch_one_window(ticker, api_token, window_start, window_end)
        if data is None:
            failed_windows.append((str(window_start.date()), str(window_end.date()), reason))
        else:
            all_bars.extend(data)
        window_start = window_end + timedelta(days=1)
    return all_bars, failed_windows


def build_clean_day_groups(raw_bars):
    """Applies the full validated quality pipeline and returns clean,
    ready-to-score per-day bar groups.

    Returns:
      clean_days: {date_str: DataFrame with minutes_from_open added}
      excluded_days: DataFrame of incomplete days (from day_completeness_report)
      flagged_moves: DataFrame of implausible-move days (needs manual review)
      outside_session_info: dict summary of auction/other bars found
    """
    if not raw_bars:
        return {}, pd.DataFrame(), pd.DataFrame(), {"n_outside_bars": 0, "dates_affected": []}

    df, err = to_sydney_and_classify(raw_bars)
    if err:
        return {}, pd.DataFrame(), pd.DataFrame(), {"error": err}

    completeness_report, complete_dates = day_completeness_report(df)
    outside_info = outside_session_summary(df)
    flagged_moves = flag_implausible_moves(df, complete_dates)
    flagged_dates = set(str(d) for d in flagged_moves["date"]) if not flagged_moves.empty else set()

    # Final clean set: complete AND not flagged as implausible.
    usable_dates = complete_dates - flagged_dates

    continuous = df[df["in_continuous_session"] & df["sydney_date"].astype(str).isin(usable_dates)].copy()
    clean_days = {}
    for date_str, day_df in continuous.groupby(continuous["sydney_date"].astype(str)):
        day_df = day_df.sort_values("sydney_dt").reset_index(drop=True)
        session_start = day_df["sydney_dt"].iloc[0]
        day_df["minutes_from_open"] = (day_df["sydney_dt"] - session_start).dt.total_seconds() / 60
        clean_days[date_str] = day_df

    excluded_days = completeness_report[completeness_report["status"] != "complete"]
    return clean_days, excluded_days, flagged_moves, outside_info
