"""
V2 DIAGNOSTICS — Tests 2 & 3
================================
Does NOT modify V1. Imports config_v1.py's frozen HYPOTHESES as-is and
reuses the same driver/outcome data pipeline. This is a separate,
additional analysis layer on top of the frozen baseline, not a new
version of it.

Test 2: confirmed-vs-unconfirmed comparison, split chronologically AND
per-ticker — is BHP ADR confirmation a validated feature or a pattern
that only existed in the data already looked at.

Test 3: opening-gap relationship — pre-specified buckets (reused from
config_v1.GAP_BUCKETS, nothing new) plus a Spearman monotonic
correlation between gap size and same-day return, as a diagnostic only.
Diagnostic set = train+validation pooled. Test set analysed separately,
touched once, never used to pick a threshold.

Scoped to the two hypothesis clusters that survived V1 (Energy H1/H1b,
Iron Ore H4/H4b), both directions — not all 18, per the V2 brief of
translating SURVIVING findings into tradeable setups.
"""

import streamlit as st
import pandas as pd

from config_v1 import DRIVERS, ASX_THEME_STOCKS, HYPOTHESES, COSTS
from historical_data import build_driver_table, align_to_asx_sessions, build_asx_outcome_tables, fetch_raw_history
from backtest import compare_confirmed_vs_unconfirmed_by_split, gap_relationship_analysis

st.set_page_config(page_title="V2 Diagnostics", layout="wide")
st.title("V2 Diagnostics — Tests 2 & 3")
st.caption("Does not modify config_v1.py or any V1 threshold. Reuses the frozen hypotheses as-is.")

CONFIRMATION_PAIRS = [("H4_long", "H4b_long"), ("H4_short", "H4b_short")]
GAP_TEST_HYPOTHESES = ["H1_long", "H1_short", "H1b_long", "H1b_short",
                        "H4_long", "H4_short", "H4b_long", "H4b_short"]
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


def needed_tickers():
    themes = {"Energy (Oil/Gas)", "Iron Ore"}
    tickers = set()
    for t in themes:
        tickers.update(ASX_THEME_STOCKS[t])
    return tuple(sorted(tickers))


def flatten_confirmation_by_split(results):
    rows = []
    for r in results:
        if "error" in r:
            rows.append({"base_id": r.get("base_id"), "error": r["error"]})
            continue
        for group in ["confirmed", "unconfirmed_only"]:
            g = r[group]
            for col in OUTCOME_COLUMNS:
                for split in ["train", "validation", "test"]:
                    stats = g["day_level"][col][split]
                    row = {"base_id": r["base_id"], "confirmed_id": r["confirmed_id"], "group": group,
                           "granularity": "day_level", "outcome_column": col, "split": split}
                    row.update(stats)
                    rows.append(row)
                for ticker, splits in g["per_stock"][col].items():
                    for split, stats in splits.items():
                        row = {"base_id": r["base_id"], "confirmed_id": r["confirmed_id"], "group": group,
                               "granularity": "per_stock", "ticker": ticker,
                               "outcome_column": col, "split": split}
                        row.update(stats)
                        rows.append(row)
    return pd.DataFrame(rows)


def flatten_gap_analysis(results):
    rows = []
    for r in results:
        if r.get("n_matched_days", 0) == 0:
            continue
        for period_key in ["diagnostic_train_validation", "held_out_test"]:
            period = r[period_key]
            rows.append({
                "hypothesis_id": r["hypothesis_id"], "period": period["label"], "n_days": period["n_days"],
                "bucket": "ALL (correlation)", "granularity": "day_level", "ticker": "",
                "spearman_corr": period["spearman_gap_vs_return_day_level"],
            })
            for ticker, corr in period["spearman_gap_vs_return_per_ticker"].items():
                rows.append({
                    "hypothesis_id": r["hypothesis_id"], "period": period["label"], "n_days": period["n_days"],
                    "bucket": "ALL (correlation)", "granularity": "per_stock", "ticker": ticker,
                    "spearman_corr": corr,
                })
            for bucket, stats in period["gap_buckets_day_level"].items():
                row = {"hypothesis_id": r["hypothesis_id"], "period": period["label"], "n_days": period["n_days"],
                       "bucket": bucket, "granularity": "day_level", "ticker": ""}
                row.update(stats)
                rows.append(row)
            for ticker, buckets in period["gap_buckets_per_ticker"].items():
                for bucket, stats in buckets.items():
                    row = {"hypothesis_id": r["hypothesis_id"], "period": period["label"], "n_days": period["n_days"],
                           "bucket": bucket, "granularity": "per_stock", "ticker": ticker}
                    row.update(stats)
                    rows.append(row)
    return pd.DataFrame(rows)


if st.button("Run V2 diagnostics (Tests 2 & 3)", type="primary", use_container_width=True):
    with st.spinner("Pulling driver history..."):
        driver_table, driver_failures = load_driver_table()
    if driver_failures:
        st.warning(f"{len(driver_failures)} driver ticker(s) failed: {', '.join(f'{n} ({t})' for n, t in driver_failures)}")

    tickers_needed = needed_tickers()
    with st.spinner(f"Pulling ASX outcome history ({len(tickers_needed)} tickers)..."):
        asx_outcomes, asx_failures = load_asx_outcomes(tickers_needed)
    if asx_failures:
        st.warning(f"{len(asx_failures)} ASX ticker(s) failed: {', '.join(asx_failures)}")

    with st.spinner("Aligning overnight drivers to ASX sessions..."):
        all_asx_dates = sorted(set().union(*[df.index for df in asx_outcomes.values() if not df.empty]))
        aligned = align_to_asx_sessions(driver_table, all_asx_dates)

    by_id = {h["id"]: h for h in HYPOTHESES}

    with st.spinner("Test 2: confirmation value, split chronologically and per ticker..."):
        confirm_results = [
            compare_confirmed_vs_unconfirmed_by_split(by_id[b], by_id[c], aligned, asx_outcomes, ASX_THEME_STOCKS, COSTS)
            for b, c in CONFIRMATION_PAIRS
        ]

    with st.spinner("Test 3: opening-gap relationship (buckets + monotonic correlation)..."):
        gap_results = [
            gap_relationship_analysis(by_id[hid], aligned, asx_outcomes, ASX_THEME_STOCKS, COSTS)
            for hid in GAP_TEST_HYPOTHESES
        ]

    st.divider()
    st.subheader("Test 2 — Confirmation value, out-of-sample")
    for r in confirm_results:
        if "error" in r:
            st.write(f"{r.get('base_id')}: {r['error']}")
            continue
        with st.expander(f"{r['base_id']} vs {r['confirmed_id']}"):
            for group in ["confirmed", "unconfirmed_only"]:
                g = r[group]
                st.markdown(f"**{group}** — {g['n_total_days']} days total "
                            f"(train {g['n_train_days']} / val {g['n_validation_days']} / test {g['n_test_days']})")
                df = pd.DataFrame(g["day_level"]["open_to_close_pct"]).T
                st.dataframe(df, use_container_width=True)
                rows = []
                for ticker, splits in g["per_stock"]["open_to_close_pct"].items():
                    for split, stats in splits.items():
                        rows.append({"ticker": ticker, "split": split, "n": stats.get("n"),
                                     "expectancy_net": stats.get("expectancy_net")})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Test 3 — Opening-gap relationship")
    for r in gap_results:
        if r.get("n_matched_days", 0) == 0:
            continue
        with st.expander(f"{r['hypothesis_id']} — {r['label']}"):
            for period_key, period_label in [("diagnostic_train_validation", "Train+Validation (diagnostic)"),
                                              ("held_out_test", "Test (checked once)")]:
                period = r[period_key]
                st.markdown(f"**{period_label}** — {period['n_days']} days")
                st.write(f"Spearman correlation (gap vs return, basket day-level): "
                         f"{period['spearman_gap_vs_return_day_level']}")
                bucket_df = pd.DataFrame(period["gap_buckets_day_level"]).T
                st.dataframe(bucket_df, use_container_width=True)

    st.divider()
    st.subheader("Downloads")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Test 2 — Confirmation by split & ticker CSV",
                            data=flatten_confirmation_by_split(confirm_results).to_csv(index=False),
                            file_name="v2_test2_confirmation_by_split.csv", mime="text/csv", use_container_width=True)
    with col2:
        st.download_button("Test 3 — Gap relationship CSV",
                            data=flatten_gap_analysis(gap_results).to_csv(index=False),
                            file_name="v2_test3_gap_relationship.csv", mime="text/csv", use_container_width=True)
else:
    st.info("Tap to run Tests 2 & 3. Uses the same cached driver/ASX data as the Phase 1 app if run within 12 hours.")
