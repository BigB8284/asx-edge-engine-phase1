"""
Streamlit wrapper for vix_threshold_rerun.py — same pattern as
sanity_check_app.py. Deploy alongside vix_threshold_rerun.py,
config_v1.py, historical_data.py, backtest.py, and
driver_threshold_sanity_check.py (it imports the approved VIX
thresholds from that last one, single source of truth).

One ticker, so this should finish in a few seconds. Disposable —
delete after downloading the results CSV.
"""
import streamlit as st
import pandas as pd

from config_v1 import DRIVERS
from historical_data import fetch_raw_history, compute_valid_pct_change, align_to_asx_sessions
from backtest import chronological_split
from vix_threshold_rerun import DRIVER_NAME, THRESHOLDS
from driver_threshold_sanity_check import ASX_CALENDAR_TICKER, MIN_TRAIN_N

st.set_page_config(page_title="VIX Threshold Rerun")
st.title("VIX Threshold Rerun")
st.caption(
    f"Narrow follow-up check: {DRIVER_NAME.upper()} only, at the widened band "
    f"{['±' + str(t) + '%' for t in THRESHOLDS]}, approved 2026-08-25 after the "
    f"original ±1/2/3% band fired on 26-45% of all days. TRAIN-only, no "
    f"validation/test touched."
)

if st.button("Run VIX rerun", type="primary"):
    status = st.empty()

    status.write(f"Building ASX trading calendar from {ASX_CALENDAR_TICKER}...")
    asx_hist = fetch_raw_history(ASX_CALENDAR_TICKER)
    if asx_hist is None or asx_hist.empty:
        st.error(f"Could not fetch {ASX_CALENDAR_TICKER} — aborting.")
        st.stop()
    asx_dates = pd.to_datetime(asx_hist.index.date)

    ticker, role, first_available, notes = DRIVERS[DRIVER_NAME]
    status.write(f"Fetching {DRIVER_NAME} ({ticker})...")
    hist = fetch_raw_history(ticker)
    if hist is None or hist.empty:
        st.error(f"Could not fetch {ticker} — aborting.")
        st.stop()

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

    status.write("Done.")
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch")
    st.download_button(
        "Download results CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="vix_threshold_rerun_results.csv",
        mime="text/csv",
    )
