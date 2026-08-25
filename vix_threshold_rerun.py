"""
VIX THRESHOLD RERUN — narrow follow-up to driver_threshold_sanity_check.py
============================================================================
The original run showed the +-1/2/3% band on VIX firing on 26-45% of all
trading days — not a selective filter. Brent approved widening it to
+-5/10/15% instead of dropping VIX outright. This script re-checks ONLY
that new band, reusing the exact same logic and data source as the main
sanity check (see driver_threshold_sanity_check.py's
PART_B_THRESHOLDS_OVERRIDE["vix"] — this script imports that value
rather than redefining it, so there is one source of truth for what the
approved VIX thresholds are).

Deliberately narrow rather than a full 34-driver rerun: the other 33
drivers' numbers already came back clean and haven't changed; there's no
reason to re-fetch them and burn time/API calls.

Same TRAIN-only, no-validation/no-test discipline as the main script.
"""
import sys
import pandas as pd

from config_v1 import DRIVERS
from historical_data import fetch_raw_history, compute_valid_pct_change, align_to_asx_sessions
from backtest import chronological_split
from driver_threshold_sanity_check import ASX_CALENDAR_TICKER, PART_B_THRESHOLDS_OVERRIDE, MIN_TRAIN_N

DRIVER_NAME = "vix"
THRESHOLDS = PART_B_THRESHOLDS_OVERRIDE[DRIVER_NAME]


def main():
    print(f"Building ASX trading calendar from {ASX_CALENDAR_TICKER}...", file=sys.stderr)
    asx_hist = fetch_raw_history(ASX_CALENDAR_TICKER)
    if asx_hist is None or asx_hist.empty:
        raise SystemExit(f"Could not fetch {ASX_CALENDAR_TICKER} to build the ASX trading calendar — aborting.")
    asx_dates = pd.to_datetime(asx_hist.index.date)

    ticker, role, first_available, notes = DRIVERS[DRIVER_NAME]
    print(f"Fetching {DRIVER_NAME} ({ticker})...", file=sys.stderr)
    hist = fetch_raw_history(ticker)
    if hist is None or hist.empty:
        raise SystemExit(f"Could not fetch {ticker} — aborting.")

    driver_series = compute_valid_pct_change(hist)
    driver_table = driver_series.to_frame(name=DRIVER_NAME)
    aligned = align_to_asx_sessions(driver_table, asx_dates)

    usable_from = pd.Timestamp(first_available)
    aligned = aligned[aligned.index >= usable_from]
    col = aligned[DRIVER_NAME].dropna()

    rows = []
    for threshold in THRESHOLDS:
        pos_dates = sorted(col[col >= threshold].index)
        neg_dates = sorted(col[col <= -threshold].index)
        pos_train, _, _ = chronological_split(pos_dates) if pos_dates else ([], [], [])
        neg_train, _, _ = chronological_split(neg_dates) if neg_dates else ([], [], [])
        rows.append({
            "driver": DRIVER_NAME,
            "ticker": ticker,
            "threshold_pct": threshold,
            "usable_from": str(usable_from.date()),
            "n_days_after_alignment": len(col),
            "positive_condition_n_total": len(pos_dates),
            "positive_condition_n_train": len(pos_train),
            "positive_rate_pct_of_days": round(len(pos_dates) / len(col) * 100, 1) if len(col) else None,
            "positive_ineligible_lt_30": len(pos_train) < MIN_TRAIN_N,
            "negative_condition_n_total": len(neg_dates),
            "negative_condition_n_train": len(neg_train),
            "negative_rate_pct_of_days": round(len(neg_dates) / len(col) * 100, 1) if len(col) else None,
            "negative_ineligible_lt_30": len(neg_train) < MIN_TRAIN_N,
        })

    df = pd.DataFrame(rows)
    out_path = "vix_threshold_rerun_results.csv"
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
