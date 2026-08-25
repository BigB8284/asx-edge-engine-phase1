"""
Streamlit wrapper for driver_threshold_sanity_check.py.

This is a run-once utility, not part of the live scanner or the pilot
app. It exists purely so the sanity check can be executed inside a
GitHub/Streamlit-only workflow (click a button, watch progress, download
a CSV) rather than needing a local Python environment.

Deploy this alongside driver_threshold_sanity_check.py, config_v1.py,
historical_data.py and backtest.py (same folder, same repo). It imports
its threshold definitions and conditions directly from
driver_threshold_sanity_check.py rather than redefining them, so there
is exactly one source of truth for what gets searched.

Safe to delete after you've downloaded the results CSV — it isn't
referenced by anything else.
"""
import streamlit as st
import pandas as pd

from config_v1 import DRIVERS
from historical_data import fetch_raw_history, compute_valid_pct_change, align_to_asx_sessions
from backtest import chronological_split
from driver_threshold_sanity_check import (
    ASX_CALENDAR_TICKER, build_all_conditions, FLAGGED_VOLATILE_DRIVERS, MIN_TRAIN_N,
)

st.set_page_config(page_title="Driver Threshold Sanity Check")
st.title("Driver Threshold Sanity Check")
st.caption(
    "One-off pre-coding check for the V3 reverse-discovery pilot. "
    "Counts TRAIN-only occurrence days per driver/threshold, both directions. "
    "Does NOT touch validation or test, and does NOT look at ASX stock outcomes — "
    "driver-side counting only."
)

if st.button("Run sanity check", type="primary"):
    progress = st.progress(0.0)
    status = st.empty()

    status.write(f"Building ASX trading calendar from {ASX_CALENDAR_TICKER}...")
    asx_hist = fetch_raw_history(ASX_CALENDAR_TICKER)
    if asx_hist is None or asx_hist.empty:
        st.error(f"Could not fetch {ASX_CALENDAR_TICKER} to build the ASX trading calendar — aborting.")
        st.stop()
    asx_dates = pd.to_datetime(asx_hist.index.date)

    conditions = build_all_conditions()
    needed_drivers = sorted(set(name for name, _, _ in conditions))

    rows = []
    failed_drivers = []
    for i, name in enumerate(needed_drivers):
        ticker, role, first_available, notes = DRIVERS[name]
        status.write(f"Fetching {name} ({ticker})...  [{i + 1}/{len(needed_drivers)}]")
        progress.progress((i + 1) / len(needed_drivers))

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

    status.write("Done.")
    if failed_drivers:
        st.warning(f"Failed to fetch (excluded from results below): {failed_drivers}")

    if not rows:
        st.error("No conditions produced any results.")
        st.stop()

    df = pd.DataFrame(rows).sort_values(["driver", "threshold_pct"]).reset_index(drop=True)
    st.success(f"Done — {len(df)} driver/threshold conditions computed.")
    st.dataframe(df, width="stretch")
    st.download_button(
        "Download results CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="driver_threshold_sanity_check_results.csv",
        mime="text/csv",
    )
