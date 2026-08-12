"""
HISTORICAL EDGE ENGINE — PHASE 1 BASELINE
=============================================
Deploy like your other tools. This is a heavier run than the validator —
~60 tickers of full daily history plus 18 hypothesis evaluations — so it
can take a few minutes on first run. Results are cached for 12 hours so
you're not re-pulling everything every time you open the app.

This produces the FROZEN V1 baseline: the first, untouched pass across
train/validation/test for every hypothesis. Once you've reviewed it and
we agree it's the baseline, don't re-run this to chase better numbers —
any future change gets compared against the downloaded CSV from this
run, in a v2 file, not by re-running this one differently.
"""

import streamlit as st
import pandas as pd

from config_v1 import DRIVERS, ASX_THEME_STOCKS, HYPOTHESES, COSTS
from historical_data import build_driver_table, align_to_asx_sessions, build_asx_outcome_tables, fetch_raw_history
from backtest import evaluate_hypothesis

st.set_page_config(page_title="Phase 1 Baseline", layout="wide")
st.title("Historical Edge Engine — Phase 1 Baseline")
st.caption("First untouched pass across all 18 hypotheses. Frozen once reviewed — future changes compare against this, they don't overwrite it.")


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def cached_fetch(ticker):
    """Caches EACH TICKER individually. If a batch pull partially fails
    and the app is re-run, tickers that already succeeded are served
    instantly from here — only the ones that actually failed get
    re-attempted, instead of re-pulling all ~59 from scratch."""
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


def flatten_for_csv(results):
    """One row per (hypothesis, outcome_column, split) — the archival
    format for the frozen baseline CSV."""
    rows = []
    for r in results:
        if r.get("n_matched", 0) == 0:
            rows.append({"hypothesis_id": r["hypothesis_id"], "note": r.get("note", "")})
            continue
        for col in ["open_to_close_pct", "next_session_return", "day2_return", "day3_return"]:
            for split in ["train", "validation", "test"]:
                stats = r[col][split]
                row = {
                    "hypothesis_id": r["hypothesis_id"], "label": r["label"],
                    "direction": r["direction"], "theme": r["theme"],
                    "outcome_column": col, "split": split,
                }
                row.update(stats)
                rows.append(row)
    return pd.DataFrame(rows)


if st.button("Run Phase 1 baseline", type="primary", use_container_width=True):
    with st.spinner("Pulling driver history (~37 tickers)..."):
        driver_table, driver_failures = load_driver_table()
    st.success(f"Driver table: {driver_table.shape[0]} rows x {driver_table.shape[1]} drivers")
    if driver_failures:
        st.warning(
            f"{len(driver_failures)} driver ticker(s) failed even after retries and were skipped "
            f"(not faked, not substituted): {', '.join(f'{n} ({t})' for n, t in driver_failures)}. "
            f"Any hypothesis using these will show a reduced or missing sample — re-running the app "
            f"later will only retry these, not re-pull everything."
        )

    tickers_needed = needed_asx_tickers()
    with st.spinner(f"Pulling ASX outcome history ({len(tickers_needed)} tickers)..."):
        asx_outcomes, asx_failures = load_asx_outcomes(tickers_needed)
    st.success(f"ASX outcomes pulled for {len(asx_outcomes)} of {len(tickers_needed)} tickers")
    if asx_failures:
        st.warning(
            f"{len(asx_failures)} ASX ticker(s) failed even after retries and were skipped: "
            f"{', '.join(asx_failures)}. Baskets containing these will run on the remaining stocks only."
        )

    with st.spinner("Aligning overnight drivers to ASX sessions (no look-ahead)..."):
        all_asx_dates = sorted(set().union(*[df.index for df in asx_outcomes.values() if not df.empty]))
        aligned = align_to_asx_sessions(driver_table, all_asx_dates)

    with st.spinner("Evaluating 18 hypotheses across train/validation/test..."):
        results = [
            evaluate_hypothesis(h, aligned, asx_outcomes, ASX_THEME_STOCKS, COSTS)
            for h in HYPOTHESES
        ]

    st.divider()
    st.subheader("Summary — open→close outcome, by hypothesis")
    summary_rows = []
    for r in results:
        if r.get("n_matched", 0) == 0:
            summary_rows.append({"id": r["hypothesis_id"], "matched": 0, "note": r.get("note")})
            continue
        val_stats = r["open_to_close_pct"]["validation"]
        test_stats = r["open_to_close_pct"]["test"]
        summary_rows.append({
            "id": r["hypothesis_id"], "direction": r["direction"], "theme": r["theme"],
            "n_matched": r["n_matched"], "n_train": r["n_train"], "n_val": r["n_validation"], "n_test": r["n_test"],
            "val_band": val_stats.get("band"), "val_win_rate": val_stats.get("win_rate"),
            "val_expectancy_net": val_stats.get("expectancy_net"),
            "test_band": test_stats.get("band"), "test_win_rate": test_stats.get("win_rate"),
            "test_expectancy_net": test_stats.get("expectancy_net"),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    st.caption("val_/test_ expectancy_net is already after 5bps round-trip commission. band reflects sample-size confidence, not a claim of a confirmed edge.")

    st.divider()
    st.subheader("Full detail per hypothesis")
    for r in results:
        label = r.get("label", r["hypothesis_id"])
        with st.expander(f"{r['hypothesis_id']} — {label} (matched: {r.get('n_matched', 0)})"):
            if r.get("n_matched", 0) == 0:
                st.write(r.get("note", "No matches"))
                continue
            for col in ["open_to_close_pct", "next_session_return", "day2_return", "day3_return"]:
                st.markdown(f"**{col}**")
                split_df = pd.DataFrame(r[col]).T
                st.dataframe(split_df, use_container_width=True)
            if r.get("gap_breakdown_validation"):
                st.markdown("**Opening-gap breakdown (validation set)**")
                gap_df = pd.DataFrame(r["gap_breakdown_validation"]).T
                st.dataframe(gap_df, use_container_width=True)

    st.divider()
    csv_df = flatten_for_csv(results)
    st.download_button(
        "Download frozen V1 baseline results (CSV)",
        data=csv_df.to_csv(index=False),
        file_name="baseline_v1_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption("This CSV is the frozen baseline. Save it alongside config_v1.py. Every future change gets compared against this file, not re-generated by re-running this app differently.")
else:
    st.info("Tap 'Run Phase 1 baseline' to pull data and evaluate all 18 hypotheses. First run takes a few minutes.")
