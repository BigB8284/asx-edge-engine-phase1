"""
HISTORICAL EDGE ENGINE — PHASE 1 BASELINE (v1.1 — instrumentation only)
===========================================================================
Same V1 hypotheses, same config_v1.py, same thresholds — NOTHING about
what counts as a match has changed. This version reports the SAME
results at finer granularity:
  - day_level: one observation per matched day (basket average) — the
    correct basis for a genuine day-count win rate / CI, since pooling
    every (stock, day) pair as independent evidence overstates
    confidence when stocks in a basket move together.
  - per_stock: each basket member's own result, to see whether an
    apparent edge is broad or concentrated in one name.
  - confirmation comparison: for the three genuinely nested pairs
    (Energy, Gold, Iron Ore), isolates what the confirming driver
    actually added using the ACTUAL matched dates, not a headline
    percentage comparison across two different samples. Lithium's
    H3b is NOT a nested confirmation of H3 (different threshold on
    the same driver, not a strict superset condition) so it's
    excluded from this comparison rather than faked.

Deploy like before. This run takes about as long as the last one.
"""

import streamlit as st
import pandas as pd

from config_v1 import DRIVERS, ASX_THEME_STOCKS, HYPOTHESES, COSTS
from historical_data import build_driver_table, align_to_asx_sessions, build_asx_outcome_tables, fetch_raw_history
from backtest import evaluate_hypothesis_detailed, compare_confirmed_vs_unconfirmed

st.set_page_config(page_title="Phase 1 Baseline", layout="wide")
st.title("Historical Edge Engine — Phase 1 Baseline (detailed)")
st.caption("Same V1 hypotheses and thresholds as before — this run adds day-level, per-stock, and confirmation-isolation detail on top of the same results.")

CONFIRMATION_PAIRS = [
    ("H1_long", "H1b_long"), ("H1_short", "H1b_short"),
    ("H2_long", "H2b_long"), ("H2_short", "H2b_short"),
    ("H4_long", "H4b_long"), ("H4_short", "H4b_short"),
]
OUTCOME_COLUMNS = ["open_to_close_pct", "next_session_return", "day2_return", "day3_return"]


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def cached_fetch(ticker):
    return fetch_raw_history(ticker)


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_driver_table():
    return build_driver_table(list(DRIVERS.keys()), fetch_fn=cached_fetch)


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_asx_outcomes(tickers_tuple):
    return build_asx_outcome_tables(list(tickers_tuple), fetch_fn=cached_fetch)


def needed_asx_tickers():
    themes_used = {h["theme"] for h in HYPOTHESES}
    tickers = set()
    for theme in themes_used:
        tickers.update(ASX_THEME_STOCKS[theme])
    return tuple(sorted(tickers))


def flatten_day_and_stockday(results):
    rows = []
    for r in results:
        if r.get("n_matched_days", 0) == 0:
            rows.append({"hypothesis_id": r["hypothesis_id"], "note": r.get("note", "")})
            continue
        for granularity in ["day_level", "stock_day"]:
            for col in OUTCOME_COLUMNS:
                for split in ["train", "validation", "test"]:
                    stats = r[granularity][col][split]
                    row = {
                        "hypothesis_id": r["hypothesis_id"], "label": r["label"],
                        "direction": r["direction"], "theme": r["theme"],
                        "granularity": granularity, "outcome_column": col, "split": split,
                        "n_matched_days": r["n_matched_days"], "n_train_days": r["n_train_days"],
                        "n_validation_days": r["n_validation_days"], "n_test_days": r["n_test_days"],
                    }
                    row.update(stats)
                    rows.append(row)
    return pd.DataFrame(rows)


def flatten_per_stock(results):
    rows = []
    for r in results:
        if r.get("n_matched_days", 0) == 0:
            continue
        for col in OUTCOME_COLUMNS:
            for ticker, splits in r["per_stock"][col].items():
                for split, stats in splits.items():
                    row = {
                        "hypothesis_id": r["hypothesis_id"], "direction": r["direction"],
                        "theme": r["theme"], "ticker": ticker,
                        "outcome_column": col, "split": split,
                    }
                    row.update(stats)
                    rows.append(row)
    return pd.DataFrame(rows)


def flatten_confirmation_comparisons(comparisons):
    rows = []
    for comp in comparisons:
        if "error" in comp:
            rows.append({"base_id": comp.get("base_id"), "error": comp["error"]})
            continue
        for group in ["confirmed", "unconfirmed_only"]:
            for col in OUTCOME_COLUMNS:
                stats = comp[group][col]
                row = {
                    "base_id": comp["base_id"], "confirmed_id": comp["confirmed_id"],
                    "n_confirmed_days": comp["n_confirmed_days"], "n_unconfirmed_only_days": comp["n_unconfirmed_only_days"],
                    "group": group, "outcome_column": col,
                }
                row.update(stats)
                rows.append(row)
    return pd.DataFrame(rows)


if st.button("Run Phase 1 baseline (detailed)", type="primary", use_container_width=True):
    with st.spinner("Pulling driver history (~37 tickers)..."):
        driver_table, driver_failures = load_driver_table()
    st.success(f"Driver table: {driver_table.shape[0]} rows x {driver_table.shape[1]} drivers")
    if driver_failures:
        st.warning(f"{len(driver_failures)} driver ticker(s) failed and were skipped: "
                   f"{', '.join(f'{n} ({t})' for n, t in driver_failures)}.")

    tickers_needed = needed_asx_tickers()
    with st.spinner(f"Pulling ASX outcome history ({len(tickers_needed)} tickers)..."):
        asx_outcomes, asx_failures = load_asx_outcomes(tickers_needed)
    st.success(f"ASX outcomes pulled for {len(asx_outcomes)} of {len(tickers_needed)} tickers")
    if asx_failures:
        st.warning(f"{len(asx_failures)} ASX ticker(s) failed and were skipped: {', '.join(asx_failures)}.")

    with st.spinner("Aligning overnight drivers to ASX sessions (no look-ahead)..."):
        all_asx_dates = sorted(set().union(*[df.index for df in asx_outcomes.values() if not df.empty]))
        aligned = align_to_asx_sessions(driver_table, all_asx_dates)

    with st.spinner("Evaluating 18 hypotheses at day-level, stock-day, and per-stock granularity..."):
        results = [
            evaluate_hypothesis_detailed(h, aligned, asx_outcomes, ASX_THEME_STOCKS, COSTS)
            for h in HYPOTHESES
        ]

    with st.spinner("Isolating confirmation value on matched dates (Energy, Gold, Iron Ore)..."):
        by_id = {h["id"]: h for h in HYPOTHESES}
        comparisons = [
            compare_confirmed_vs_unconfirmed(by_id[base_id], by_id[conf_id], aligned, asx_outcomes, ASX_THEME_STOCKS, COSTS)
            for base_id, conf_id in CONFIRMATION_PAIRS
        ]

    st.divider()
    st.subheader("Day-level summary — open→close, by hypothesis")
    st.caption("day_count is the TRUE independent sample size. win_rate/expectancy here are basket-average-per-day, not pooled stock-days.")
    summary_rows = []
    for r in results:
        if r.get("n_matched_days", 0) == 0:
            summary_rows.append({"id": r["hypothesis_id"], "days": 0, "note": r.get("note")})
            continue
        val = r["day_level"]["open_to_close_pct"]["validation"]
        test = r["day_level"]["open_to_close_pct"]["test"]
        summary_rows.append({
            "id": r["hypothesis_id"], "direction": r["direction"], "theme": r["theme"],
            "n_train_days": r["n_train_days"], "n_val_days": r["n_validation_days"], "n_test_days": r["n_test_days"],
            "val_band": val.get("band"), "val_win_rate": val.get("win_rate"), "val_expectancy_net": val.get("expectancy_net"),
            "test_band": test.get("band"), "test_win_rate": test.get("win_rate"), "test_expectancy_net": test.get("expectancy_net"),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Confirmation isolation — does the confirming driver add anything, on matched dates")
    for comp in comparisons:
        if "error" in comp:
            st.write(f"{comp.get('base_id')}: {comp['error']}")
            continue
        with st.expander(f"{comp['base_id']} vs {comp['confirmed_id']} — confirmed: {comp['n_confirmed_days']} days, unconfirmed-only: {comp['n_unconfirmed_only_days']} days"):
            for col in OUTCOME_COLUMNS:
                st.markdown(f"**{col}**")
                df = pd.DataFrame({"confirmed": comp["confirmed"][col], "unconfirmed_only": comp["unconfirmed_only"][col]}).T
                st.dataframe(df, use_container_width=True)

    st.divider()
    st.subheader("Per-stock breakdown")
    for r in results:
        if r.get("n_matched_days", 0) == 0:
            continue
        with st.expander(f"{r['hypothesis_id']} — per-stock ({', '.join(r['tickers'])})"):
            for col in OUTCOME_COLUMNS:
                st.markdown(f"**{col}** — validation win_rate / expectancy_net by stock")
                rows = []
                for ticker, splits in r["per_stock"][col].items():
                    val = splits["validation"]
                    rows.append({"ticker": ticker, "n": val.get("n"), "band": val.get("band"),
                                 "win_rate": val.get("win_rate"), "expectancy_net": val.get("expectancy_net")})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Downloads")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("Day-level + stock-day CSV", data=flatten_day_and_stockday(results).to_csv(index=False),
                            file_name="baseline_v1_daylevel_and_stockday.csv", mime="text/csv", use_container_width=True)
    with col2:
        st.download_button("Per-stock CSV", data=flatten_per_stock(results).to_csv(index=False),
                            file_name="baseline_v1_per_stock.csv", mime="text/csv", use_container_width=True)
    with col3:
        st.download_button("Confirmation comparison CSV", data=flatten_confirmation_comparisons(comparisons).to_csv(index=False),
                            file_name="baseline_v1_confirmation_comparison.csv", mime="text/csv", use_container_width=True)
    st.caption("These three files plus the original baseline_v1_results.csv together are the complete frozen V1 baseline.")
else:
    st.info("Tap to run. Same hypotheses as before, richer output — takes a similar amount of time to the last run.")
