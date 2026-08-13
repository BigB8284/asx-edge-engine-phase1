"""
PHASE A — Pipeline validation on existing Grade-A signals
================================================================
Scope: Uranium LONG/SHORT (H6) and Iron Ore SHORT plain + confirmed
(H4_short, H4b_short) — the four hypotheses already validated at the
daily level in V1/V2. This is a PIPELINE VALIDATION exercise, not a
new research round on these themes — their existing daily grades are
frozen and untouched by anything here.

For each stock, computes the full intraday outcome profile (1/2/3/5%
thresholds, MFE/MAE, timing) on signal days, split chronologically
train/validation/test, and compares against that stock's own
unconditional (baseline) behaviour across all its clean trading days.

Uses config_v2's hypotheses and driver data unchanged. Uses the
already-validated eodhd_logic.py quality checks unchanged. Nothing
here re-implements or re-tunes anything from V1/V2.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timezone

from config_v2 import DRIVERS, ASX_THEME_STOCKS, HYPOTHESES
from historical_data import build_driver_table, align_to_asx_sessions, fetch_raw_history
from backtest import chronological_split
from intraday_data import fetch_full_intraday_history, build_clean_day_groups
from intraday_engine import compute_day_outcomes, THRESHOLDS, CHECKPOINTS, MFE_MAE_WINDOWS
from intraday_stats import aggregate_outcomes, compute_baseline_delta, format_summary_line

st.set_page_config(page_title="Phase A — Intraday Pipeline Validation", layout="wide")
st.title("Phase A — Intraday Pipeline Validation")
st.caption("Grade-A signals only (Uranium LONG/SHORT, Iron Ore SHORT plain + confirmed). Pipeline validation, not a new optimisation round.")

GRADE_A_IDS = ["H6_long", "H6_short", "H4_short", "H4b_short"]
by_id = {h["id"]: h for h in HYPOTHESES}
INTRADAY_START = datetime(2020, 10, 12, tzinfo=timezone.utc)  # confirmed earliest complete day across all 4 tickers tested

try:
    EODHD_TOKEN = st.secrets["EODHD_API_TOKEN"]
except Exception:
    st.error("No EODHD_API_TOKEN found in Streamlit secrets.")
    st.stop()


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def cached_fetch_daily(ticker):
    return fetch_raw_history(ticker)


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_driver_table():
    return build_driver_table(list(DRIVERS.keys()), fetch_fn=cached_fetch_daily, drivers_lookup=DRIVERS)


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_intraday_for_ticker(ticker):
    end = datetime.now(timezone.utc)
    raw_bars, failed_windows = fetch_full_intraday_history(ticker, EODHD_TOKEN, INTRADAY_START, end)
    clean_days, excluded_days, flagged_moves, outside_info = build_clean_day_groups(raw_bars)
    return clean_days, excluded_days, flagged_moves, outside_info, failed_windows


def needed_tickers():
    themes = {by_id[i]["theme"] for i in GRADE_A_IDS}
    tickers = set()
    for t in themes:
        tickers.update(ASX_THEME_STOCKS[t])
    return sorted(tickers)


def outcomes_for_dates(clean_days, date_strs, direction):
    """clean_days keys and date_strs must both be 'YYYY-MM-DD' strings —
    the exact type mismatch that broke the exclusion logic earlier in
    this project is guarded against explicitly here."""
    outcomes = []
    for ds in date_strs:
        if ds in clean_days:
            o = compute_day_outcomes(clean_days[ds], direction)
            if o is not None:
                outcomes.append(o)
    return outcomes


def flatten_results(all_results):
    rows = []
    for r in all_results:
        for t in THRESHOLDS:
            row = {
                "hypothesis_id": r["hypothesis_id"], "ticker": r["ticker"], "direction": r["direction"],
                "split": r["split"], "n_signal_days": r["signal_agg"]["n_days"],
                "n_baseline_days": r["baseline_agg"]["n_days"], "threshold_pct": t,
                "signal_prob_by_11:00": r["delta"]["checkpoints"]["11:00"][t]["signal_probability"],
                "baseline_prob_by_11:00": r["delta"]["checkpoints"]["11:00"][t]["baseline_probability"],
                "delta_pp_by_11:00": r["delta"]["checkpoints"]["11:00"][t]["delta_pp"],
                "signal_prob_full_session": r["delta"]["checkpoints"]["full_session"][t]["signal_probability"],
                "baseline_prob_full_session": r["delta"]["checkpoints"]["full_session"][t]["baseline_probability"],
                "delta_pp_full_session": r["delta"]["checkpoints"]["full_session"][t]["delta_pp"],
                "median_time_to_threshold_min": r["signal_agg"]["thresholds"][t]["median_time_to_threshold_min"],
                "median_mae_before_reached": r["signal_agg"]["thresholds"][t]["median_mae_before_reached"],
            }
            rows.append(row)
        for w in MFE_MAE_WINDOWS:
            rows.append({
                "hypothesis_id": r["hypothesis_id"], "ticker": r["ticker"], "direction": r["direction"],
                "split": r["split"], "window": str(w), "metric": "MFE_MAE",
                "signal_median_mfe": r["signal_agg"]["windows"][w]["median_mfe"],
                "signal_median_mae": r["signal_agg"]["windows"][w]["median_mae"],
                "baseline_median_mfe": r["baseline_agg"]["windows"][w]["median_mfe"],
                "baseline_median_mae": r["baseline_agg"]["windows"][w]["median_mae"],
                "signal_median_time_to_mfe_min": r["signal_agg"]["windows"][w]["median_time_to_mfe_min"],
            })
    return pd.DataFrame(rows)


if st.button("Run Phase A", type="primary", use_container_width=True):
    with st.spinner("Pulling driver history..."):
        driver_table, driver_failures = load_driver_table()
    if driver_failures:
        st.warning(f"{len(driver_failures)} driver ticker(s) failed: {driver_failures}")

    tickers = needed_tickers()
    st.info(f"Tickers needed: {tickers}")

    ticker_data = {}
    for t in tickers:
        with st.spinner(f"Pulling full intraday history for {t} (this can take a while)..."):
            clean_days, excluded, flagged, outside, failed = load_intraday_for_ticker(t)
        ticker_data[t] = {"clean_days": clean_days, "excluded": excluded, "flagged": flagged, "outside": outside, "failed": failed}
        n_flagged = len(flagged) if hasattr(flagged, "__len__") else 0
        st.write(f"**{t}**: {len(clean_days)} clean days | {len(excluded)} excluded (incomplete) | {n_flagged} flagged (implausible move) | {len(failed)} failed fetch windows")
        if failed:
            st.warning(f"{t}: failed windows: {failed}")

    all_asx_dates = sorted(set().union(*[
        set(pd.Timestamp(d) for d in td["clean_days"].keys()) for td in ticker_data.values() if td["clean_days"]
    ]))
    if not all_asx_dates:
        st.error("No clean intraday days found across any ticker — stopping.")
        st.stop()

    with st.spinner("Aligning overnight drivers to ASX sessions..."):
        aligned = align_to_asx_sessions(driver_table, all_asx_dates)

    st.divider()
    all_results = []
    for hid in GRADE_A_IDS:
        h = by_id[hid]
        direction = h["direction"]
        theme = h["theme"]
        theme_tickers = ASX_THEME_STOCKS[theme]
        usable_from = pd.Timestamp(h["usable_from"])
        driver_slice = aligned[aligned.index >= usable_from]
        matched_dates = [d for d in driver_slice.index if h["condition"](driver_slice.loc[d])]
        train_d, val_d, test_d = chronological_split(matched_dates)

        st.subheader(f"{hid} — {h['label']}")
        st.caption(f"{len(matched_dates)} matched signal days (daily-level, frozen from V1/V2) — train {len(train_d)} / val {len(val_d)} / test {len(test_d)}")

        for ticker in theme_tickers:
            clean_days = ticker_data.get(ticker, {}).get("clean_days", {})
            if not clean_days:
                st.write(f"{ticker}: no clean intraday data — skipped")
                continue

            baseline_outcomes = [compute_day_outcomes(bars, direction) for bars in clean_days.values()]
            baseline_agg = aggregate_outcomes(baseline_outcomes)

            for split_name, split_dates in [("train", train_d), ("validation", val_d), ("test", test_d)]:
                split_date_strs = [str(d.date()) for d in split_dates]
                signal_outcomes = outcomes_for_dates(clean_days, split_date_strs, direction)
                if not signal_outcomes:
                    continue
                signal_agg = aggregate_outcomes(signal_outcomes)
                delta = compute_baseline_delta(signal_agg, baseline_agg)
                all_results.append({"hypothesis_id": hid, "ticker": ticker, "direction": direction,
                                    "split": split_name, "signal_agg": signal_agg,
                                    "baseline_agg": baseline_agg, "delta": delta})

                with st.expander(f"{ticker} [{split_name}] — {signal_agg['n_days']} signal days vs {baseline_agg['n_days']} baseline days"):
                    st.write(format_summary_line(ticker, direction, signal_agg, baseline_agg, delta))
                    rows = []
                    for t in THRESHOLDS:
                        d = delta["checkpoints"]["11:00"][t]
                        d_full = delta["checkpoints"]["full_session"][t]
                        rows.append({"threshold": f"+{t:.0f}%", "signal_by_11:00": f"{d['signal_probability']:.0f}%",
                                    "baseline_by_11:00": f"{d['baseline_probability']:.0f}%", "delta_pp": d["delta_pp"],
                                    "signal_full_session": f"{d_full['signal_probability']:.0f}%",
                                    "baseline_full_session": f"{d_full['baseline_probability']:.0f}%"})
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Download full results")
    if all_results:
        csv_df = flatten_results(all_results)
        st.download_button("Phase A full results CSV", data=csv_df.to_csv(index=False),
                            file_name="phase_a_results.csv", mime="text/csv", use_container_width=True)
else:
    st.info("Tap to run Phase A. This pulls full intraday history for 8 tickers — expect this to take a while on first run.")
