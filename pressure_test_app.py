"""
FINAL PRESSURE TEST
=======================
Runs the three genuinely new checks (leave-one-out, regime stability,
and gap behaviour for themes not yet gap-tested) across the 11
hypotheses currently surviving or being considered for live use. Uses
the original frozen thresholds from config_v2.py — nothing is
optimised, searched, or tuned here.

Scope (per the pressure-test brief):
  Uranium LONG+SHORT, Iron Ore LONG+SHORT + confirmed variants,
  Energy SHORT, REIT SHORT, Coal SHORT + Coal LONG (marginal case,
  included to let the pressure test decide rather than pre-judging),
  Lithium confirmed LONG (marginal case, same reasoning).
Explicitly excluded: everything already rejected in prior rounds
(Gold, Copper, Technology, Financials-thematic, Rare Earths, REIT
LONG, Energy confirmed, Lithium confirmed SHORT) — not retested here.
"""

import streamlit as st
import pandas as pd

from config_v2 import DRIVERS, ASX_THEME_STOCKS, HYPOTHESES, COSTS
from historical_data import build_driver_table, align_to_asx_sessions, build_asx_outcome_tables, fetch_raw_history
from backtest import leave_one_out_analysis, regime_stability_analysis, gap_relationship_analysis

st.set_page_config(page_title="Final Pressure Test", layout="wide")
st.title("Final Pressure Test")
st.caption("Leave-one-out, regime stability, and gap behaviour for the 11 surviving/considered hypotheses. Original frozen thresholds — nothing tuned.")

TARGET_IDS = ["H6_long", "H6_short", "H4_long", "H4_short", "H4b_long", "H4b_short",
              "H1_short", "H12_short", "H8_short", "H8_long", "H3c_long"]
GAP_TEST_NEEDED = ["H6_long", "H6_short", "H12_short", "H8_short", "H8_long", "H3c_long"]

by_id = {h["id"]: h for h in HYPOTHESES}
missing = [i for i in TARGET_IDS if i not in by_id]
if missing:
    st.error(f"These target IDs aren't in config_v2.HYPOTHESES: {missing} — check config_v2.py wasn't changed.")


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def cached_fetch(ticker):
    return fetch_raw_history(ticker)


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_driver_table():
    return build_driver_table(list(DRIVERS.keys()), fetch_fn=cached_fetch, drivers_lookup=DRIVERS)


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_asx_outcomes(tickers_tuple):
    return build_asx_outcome_tables(list(tickers_tuple), fetch_fn=cached_fetch)


def needed_tickers():
    themes = {by_id[i]["theme"] for i in TARGET_IDS if i in by_id}
    tickers = set()
    for t in themes:
        tickers.update(ASX_THEME_STOCKS[t])
    return tuple(sorted(tickers))


def flatten_loo(results):
    rows = []
    for r in results:
        if r.get("n_matched_days", 0) == 0:
            continue
        for split, stats in r["full_basket"].items():
            row = {"hypothesis_id": r["hypothesis_id"], "cut": "full_basket", "excluded_ticker": "",
                   "split": split}
            row.update(stats)
            rows.append(row)
        for excluded, splits in r.get("leave_one_out", {}).items():
            for split, stats in splits.items():
                row = {"hypothesis_id": r["hypothesis_id"], "cut": "leave_one_out", "excluded_ticker": excluded,
                       "split": split}
                row.update(stats)
                rows.append(row)
    return pd.DataFrame(rows)


def flatten_regime(results):
    rows = []
    for r in results:
        if r.get("n_matched_days", 0) == 0:
            continue
        for p in r["periods"]:
            row = {"hypothesis_id": r["hypothesis_id"], "period_index": p["period_index"],
                   "start_date": p["start_date"], "end_date": p["end_date"], "n_days": p["n_days"],
                   "consistent_across_all_periods": r["consistent_across_all_periods"]}
            row.update(p["stats"])
            rows.append(row)
    return pd.DataFrame(rows)


def flatten_gap(results):
    rows = []
    for r in results:
        if r.get("n_matched_days", 0) == 0:
            continue
        for period_key in ["diagnostic_train_validation", "held_out_test"]:
            period = r[period_key]
            rows.append({"hypothesis_id": r["hypothesis_id"], "period": period["label"], "n_days": period["n_days"],
                        "bucket": "ALL (correlation)", "spearman_corr": period["spearman_gap_vs_return_day_level"]})
            for bucket, stats in period["gap_buckets_day_level"].items():
                row = {"hypothesis_id": r["hypothesis_id"], "period": period["label"], "n_days": period["n_days"],
                       "bucket": bucket}
                row.update(stats)
                rows.append(row)
    return pd.DataFrame(rows)


if st.button("Run final pressure test", type="primary", use_container_width=True):
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

    with st.spinner("Leave-one-out robustness..."):
        loo_results = [leave_one_out_analysis(by_id[i], aligned, asx_outcomes, ASX_THEME_STOCKS, COSTS) for i in TARGET_IDS]

    with st.spinner("Regime stability over time..."):
        regime_results = [regime_stability_analysis(by_id[i], aligned, asx_outcomes, ASX_THEME_STOCKS, COSTS) for i in TARGET_IDS]

    with st.spinner("Gap behaviour for themes not yet checked..."):
        gap_results = [gap_relationship_analysis(by_id[i], aligned, asx_outcomes, ASX_THEME_STOCKS, COSTS) for i in GAP_TEST_NEEDED]

    st.divider()
    st.subheader("Leave-one-out — does the result survive removing any single stock?")
    for r in loo_results:
        if r.get("n_matched_days", 0) == 0:
            continue
        with st.expander(f"{r['hypothesis_id']} — {r['label']}"):
            full_val = r["full_basket"]["validation"]["expectancy_net"]
            full_test = r["full_basket"]["test"]["expectancy_net"]
            st.write(f"Full basket: validation={full_val:+.3f}%, test={full_test:+.3f}%")
            rows = []
            for excluded, splits in r["leave_one_out"].items():
                rows.append({"excluded": excluded,
                            "validation_without": splits["validation"]["expectancy_net"],
                            "test_without": splits["test"]["expectancy_net"]})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Regime stability — is this concentrated in one historical period?")
    for r in regime_results:
        if r.get("n_matched_days", 0) == 0:
            continue
        with st.expander(f"{r['hypothesis_id']} — consistent across all periods: {r['consistent_across_all_periods']}"):
            rows = []
            for p in r["periods"]:
                rows.append({"period": p["period_index"], "dates": f"{p['start_date']} to {p['end_date']}",
                            "n_days": p["n_days"], "expectancy_net": p["stats"].get("expectancy_net")})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Gap behaviour (themes not previously checked)")
    for r in gap_results:
        if r.get("n_matched_days", 0) == 0:
            continue
        with st.expander(f"{r['hypothesis_id']} — {r['label']}"):
            for period_key, period_label in [("diagnostic_train_validation", "Train+Validation"), ("held_out_test", "Test")]:
                period = r[period_key]
                st.write(f"{period_label} ({period['n_days']} days) — Spearman: {period['spearman_gap_vs_return_day_level']}")
                st.dataframe(pd.DataFrame(period["gap_buckets_day_level"]).T, use_container_width=True)

    st.divider()
    st.subheader("Downloads")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Leave-one-out CSV", data=flatten_loo(loo_results).to_csv(index=False),
                            file_name="pressure_test_leave_one_out.csv", mime="text/csv", use_container_width=True)
    with c2:
        st.download_button("Regime stability CSV", data=flatten_regime(regime_results).to_csv(index=False),
                            file_name="pressure_test_regime_stability.csv", mime="text/csv", use_container_width=True)
    with c3:
        st.download_button("Gap behaviour CSV", data=flatten_gap(gap_results).to_csv(index=False),
                            file_name="pressure_test_gap_behaviour.csv", mime="text/csv", use_container_width=True)
else:
    st.info(f"Tap to run. {len(TARGET_IDS)} hypotheses under test.")
