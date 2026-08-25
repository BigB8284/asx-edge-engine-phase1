"""
DRIVER THRESHOLD SANITY CHECK — V3 reverse-discovery pre-check
================================================================
Answers exactly one question per proposed driver/threshold condition:
does it occur often enough historically, on TRAIN, to be worth
searching? This is NOT an opportunity gate, does NOT touch VALIDATION
or TEST, and does NOT look at ASX stock outcomes at all — it only
counts driver-side condition days. No thresholds are chosen, adjusted,
or optimised here; every threshold is either an existing V1 precedent
or the flat +-1/2/3% Brent approved for drivers without one.

Reuses the project's own data-fetch and split logic exactly, rather
than reimplementing it:
  - historical_data.fetch_raw_history        (yfinance pull, retry/backoff)
  - historical_data.compute_valid_pct_change  (gap-null protected % change)
  - historical_data.align_to_asx_sessions     (driver date -> ASX session date)
  - backtest.chronological_split              (position-based train/val/test,
                                                same SPLIT_RATIOS as config_v1)

KNOWN LIMITATION — flagged, not hidden:
This counts condition-days against an ASX DAILY trading calendar
(pulled from a reference ASX ticker via yfinance), matching the
Phase 1/B convention in backtest.py. The live V3 checkpoint engine
(intraday_engine.py) runs on 5-minute EODHD intraday bars, whose
historical depth for the 16 pilot stocks has NOT been confirmed to
this script. If that intraday history starts later than ASX daily
history, the REAL usable TRAIN sample in the live reverse-discovery
run could be smaller than what's reported here. Treat these counts as
an upper bound / rarity check, not a guarantee of final usable n.

This script has not been run against live data in the environment that
wrote it (no network access to Yahoo Finance from that sandbox) — it
has only been exercised against synthetic data to confirm it runs
end-to-end without error. Run it for real in the GitHub/Streamlit
environment and send the resulting CSV back.

CHANGELOG — 2026-08-25, after the first real run
(driver_threshold_sanity_check_results.csv, 88 conditions):
  - natgas: DROPPED ENTIRELY. Not a driver relevant to Brent's ASX
    trading universe — not searched, not optimised, not included.
  - dxy +-3%: DROPPED. Zero occurrences either direction across the
    full ~35-year aligned window (0/0) — a real result, not a
    judgement call.
  - Six conditions/sides dropped for TRAIN n<30 (the n>=30 rule was
    NOT lowered to accommodate them): audusd +-2%, audusd +-3%,
    dxy +-2%, xlc +-3%, xlp +-3%, xlre +-3%.
  - vix: the flat +-1/2/3% band is REPLACED with +-5/10/15%. Real data
    showed +-1% VIX fired on 38.8%/45.3% of all days (up/down), and
    even +-3% still fired on 26.3%/30.1% — not a selective filter.
    Awaiting a fresh occurrence-count rerun (VIX only, see
    vix_threshold_rerun.py) before treating the new band as final.
"""
import sys
import pandas as pd

from config_v1 import DRIVERS
from historical_data import fetch_raw_history, compute_valid_pct_change, align_to_asx_sessions
from backtest import chronological_split

# ---------------------------------------------------------------------------
# Reference ASX trading calendar. align_to_asx_sessions only needs the
# SET of ASX trading dates, not this ticker's own price data — any
# long-history, reliably-traded ASX ticker works. CBA.AX has one of the
# deepest, cleanest histories on yfinance among the 16 pilot names.
# ---------------------------------------------------------------------------
ASX_CALENDAR_TICKER = "CBA.AX"

# ---------------------------------------------------------------------------
# PART A — existing V1 precedent thresholds, taken directly from
# config_v1.HYPOTHESES (H1-H5 pairs). LIT carries two genuine
# precedents (H3's 3.0% and H3b's 2.0%) and both are searched.
# ---------------------------------------------------------------------------
PART_A_THRESHOLDS = {
    "brent": [2.0],
    "gold": [2.0],
    "lit": [3.0, 2.0],
    "iron_ore": [1.0],
    "sp500": [1.0],
    "xle": [1.0],
    "gdx": [2.0],
    "nasdaq": [1.0],
    "bhp_adr": [1.0],
}

# ---------------------------------------------------------------------------
# PART B — remaining PRIMARY drivers with no existing V1 threshold.
# FALLBACK drivers (vale_adr, cliffs) are excluded per Brent's approval
# — config_v1.py marks them confirmation-only, not standalone.
# natgas removed 2026-08-25 (see changelog) — not relevant to Brent's
# ASX trading universe.
# ---------------------------------------------------------------------------
PART_B_DRIVERS = [
    "dow", "russell2000", "vix", "xlf", "xlv", "xly", "xlp", "xli", "xlk",
    "xlre", "xlu", "xlc", "wti", "silver", "copper", "albemarle",
    "sqm", "ura", "cameco", "uec", "coal", "audusd", "dxy", "rio_adr", "newmont",
]
PART_B_THRESHOLDS_DEFAULT = [1.0, 2.0, 3.0]

# Per-driver overrides to the default +-1/2/3% band. vix added
# 2026-08-25 (see changelog) — the default band occurred on 26-45% of
# all days in the real run, not a selective filter. These specific
# values are pending a fresh occurrence-count rerun before being
# treated as final (vix_threshold_rerun.py).
PART_B_THRESHOLDS_OVERRIDE = {
    "vix": [5.0, 10.0, 15.0],
}

# Specific (driver, threshold) conditions dropped after the real
# 2026-08-25 run — either zero/near-zero occurrence, or TRAIN n<30 on
# at least one side, per Brent's explicit review of that run's numbers.
# The n>=30 rule itself was NOT lowered to accommodate these; the
# conditions were dropped instead. See changelog above for the exact
# real counts that drove each of these.
EXCLUDED_CONDITIONS = {
    ("dxy", 3.0),
    ("dxy", 2.0),
    ("audusd", 2.0),
    ("audusd", 3.0),
    ("xlc", 3.0),
    ("xlp", 3.0),
    ("xlre", 3.0),
}

# Drivers flagged as likely too volatile for a flat +-1/2/3% band to be
# selective. natgas confirmed and dropped entirely (see changelog); vix
# confirmed too loose at +-1/2/3% and widened to +-5/10/15% — kept
# flagged here until the widened band is itself confirmed selective.
FLAGGED_VOLATILE_DRIVERS = {"vix"}

MIN_TRAIN_N = 30


def build_all_conditions():
    conditions = []
    for name, thresholds in PART_A_THRESHOLDS.items():
        for th in thresholds:
            conditions.append((name, th, "existing"))
    for name in PART_B_DRIVERS:
        thresholds = PART_B_THRESHOLDS_OVERRIDE.get(name, PART_B_THRESHOLDS_DEFAULT)
        for th in thresholds:
            conditions.append((name, th, "new"))
    conditions = [c for c in conditions if (c[0], c[1]) not in EXCLUDED_CONDITIONS]
    return conditions


def main():
    print(f"Building ASX trading calendar from {ASX_CALENDAR_TICKER}...", file=sys.stderr)
    asx_hist = fetch_raw_history(ASX_CALENDAR_TICKER)
    if asx_hist is None or asx_hist.empty:
        raise SystemExit(
            f"Could not fetch {ASX_CALENDAR_TICKER} to build the ASX trading calendar — "
            f"aborting rather than fabricating one. Try a different ASX_CALENDAR_TICKER."
        )
    asx_dates = pd.to_datetime(asx_hist.index.date)

    conditions = build_all_conditions()
    needed_drivers = sorted(set(name for name, _, _ in conditions))

    rows = []
    failed_drivers = []
    for name in needed_drivers:
        ticker, role, first_available, notes = DRIVERS[name]
        print(f"Fetching {name} ({ticker})...", file=sys.stderr)
        hist = fetch_raw_history(ticker)
        if hist is None or hist.empty:
            failed_drivers.append((name, ticker))
            continue

        driver_series = compute_valid_pct_change(hist)
        driver_table = driver_series.to_frame(name=name)
        aligned = align_to_asx_sessions(driver_table, asx_dates)

        usable_from = pd.Timestamp(first_available)
        aligned = aligned[aligned.index >= usable_from]
        col = aligned[name].dropna()

        for driver_name, threshold, source in [c for c in conditions if c[0] == name]:
            pos_dates = sorted(col[col >= threshold].index)
            neg_dates = sorted(col[col <= -threshold].index)
            pos_train, _, _ = chronological_split(pos_dates) if pos_dates else ([], [], [])
            neg_train, _, _ = chronological_split(neg_dates) if neg_dates else ([], [], [])
            rows.append({
                "driver": name,
                "ticker": ticker,
                "threshold_pct": threshold,
                "source": source,
                "usable_from": str(usable_from.date()),
                "n_days_after_alignment": len(col),
                "positive_condition_n_total": len(pos_dates),
                "positive_condition_n_train": len(pos_train),
                "positive_ineligible_lt_30": len(pos_train) < MIN_TRAIN_N,
                "negative_condition_n_total": len(neg_dates),
                "negative_condition_n_train": len(neg_train),
                "negative_ineligible_lt_30": len(neg_train) < MIN_TRAIN_N,
                "flagged_volatility_concern": name in FLAGGED_VOLATILE_DRIVERS,
            })

    if not rows:
        raise SystemExit("No conditions produced any results — check driver fetches above.")

    df = pd.DataFrame(rows).sort_values(["driver", "threshold_pct"]).reset_index(drop=True)
    out_path = "driver_threshold_sanity_check_results.csv"
    df.to_csv(out_path, index=False)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))

    if failed_drivers:
        print("\nFAILED TO FETCH (excluded from results above, not silently dropped):", file=sys.stderr)
        for name, ticker in failed_drivers:
            print(f"  {name} ({ticker})", file=sys.stderr)

    print(f"\nSaved: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
