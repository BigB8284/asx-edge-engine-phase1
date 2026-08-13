"""
V2 SCANNER — RESEARCH RUN
=============================
Runs the full V2 hypothesis set (config_v2.py): frozen Iron Ore, Energy
and Lithium hypotheses carried forward unchanged, plus the new Gold
(corrected basket), Uranium, Copper, Coal, Rare Earths, Technology,
Financials, and REITs hypotheses.

Reuses the exact same evaluate_hypothesis_detailed engine from
backtest.py — nothing about the statistics, the chronological split,
the cost model, or the day-level/per-stock granularity has changed.
This is a bigger hypothesis set through the same, already-tested
pipeline, not a new pipeline.

Every result table and CSV export carries two extra columns —
`status` and `driver_sign_convention` — populated wherever config_v2.py
sets them (currently: status=experimental on Coal, inverted sign on
REITs) and blank otherwise, so neither can get lost or mistaken for a
standard result once this data leaves the app.

First run will take longer than Phase 1 — roughly 78 tickers total
(39 drivers, ~39 ASX names) rather than ~59. Same caching, same
retry/skip-on-failure resilience as before.
"""

import streamlit as st
import pandas as pd

from config_v2 import DRIVERS, ASX_THEME_STOCKS, HYPOTHESES, COSTS
from historical_data import build_driver_table, align_to_asx_sessions, build_asx_outcome_tables, fetch_raw_history
from backtest import evaluate_hypothesis_detailed

st.set_page_config(page_title="V2 Scanner Research", layout="wide")
st.title("V2 Scanner — Research Run")
st.caption("Frozen Iron Ore/Energy/Lithium + new Gold/Uranium/Copper/Coal/Rare Earths/Tech/Financials/REITs. Same engine as Phase 1.")

OUTCOME_COLUMNS = ["open_to_close_pct", "next_session_return", "day2_return", "day3_return"]
by_id = {h["id"]: h for h in HYPOTHESES}


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def cached_fetch(ticker):
    return fetch_raw_history(ticker)


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_driver_table():
    return build_driver_table(list(DRIVERS.keys()), fetch_fn=cached_fetch, drivers_lookup=DRIVERS)


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_asx_outcomes(tickers_tuple):
    return build_asx_outcome_tables(list(tickers_tuple), fetch_fn=cached_fetch)


def needed_asx_tickers():
    themes_used = {h["theme"] for h in HYPOTHESES}
    tickers = set()
    for theme in themes_used:
        tickers.update(ASX_THEME_STOCKS[theme])
    return tuple(sorted(tickers))


def tags_for(hid):
    h = by_id[hid]
    return h.get("status", ""), h.get("driver_sign_convention", "")


def flatten_results(results):
    rows = []
    for r in results:
        status, sign_note = tags_for(r["hypothesis_id"])
        if r.get("n_matched_days", 0) == 0:
            rows.append({"hypothesis_id": r["hypothesis_id"], "status": status,
                         "driver_sign_convention": sign_note, "note": r.get("note", "")})
            continue
        for granularity in ["day_level", "stock_day"]:
            for col in OUTCOME_COLUMNS:
                for split in ["train", "validation", "test"]:
                    stats = r[granularity][col][split]
                    row = {
                        "hypothesis_id": r["hypothesis_id"], "label": r["label"], "direction": r["direction"],
                        "theme": r["theme"], "status": status, "driver_sign_convention": sign_note,
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
        status, sign_note = tags_for(r["hypothesis_id"])
        for col in OUTCOME_COLUMNS:
            for ticker, splits in r["per_stock"][col].items():
                for split, stats in splits.items():
                    row = {"hypothesis_id": r["hypothesis_id"], "direction": r["direction"], "theme": r["theme"],
                           "status": status, "driver_sign_convention": sign_note,
                           "ticker": ticker, "outcome_column": col, "split": split}
                    row.update(stats)
                    rows.append(row)
    return pd.DataFrame(rows)


if st.button("Run V2 research (all 34 hypotheses)", type="primary", use_container_width=True):
    with st.spinner(f"Pulling driver history (~{len(DRIVERS)} tickers)..."):
        driver_table, driver_failures = load_driver_table()
    st.success(f"Driver table: {driver_table.shape[0]} rows x {driver_table.shape[1]} drivers")
    if driver_failures:
        st.warning(f"{len(driver_failures)} driver ticker(s) failed and were skipped: "
                   f"{', '.join(f'{n} ({t})' for n, t in driver_failures)}")

    # Sanity-check the ^TNX scale before anything downstream trusts it
    if "us10y_yield" in driver_table.columns and not driver_table["us10y_yield"].dropna().empty:
        st.info("^TNX driver loaded — this is a % CHANGE series, not the raw yield level, "
                "so there's no direct 'does it look like 4.5%' check available here. "
                "If REIT results look implausible later, that's the first thing to re-verify.")

    tickers_needed = needed_asx_tickers()
    with st.spinner(f"Pulling ASX outcome history (~{len(tickers_needed)} tickers)..."):
        asx_outcomes, asx_failures = load_asx_outcomes(tickers_needed)
    st.success(f"ASX outcomes pulled for {len(asx_outcomes)} of {len(tickers_needed)} tickers")
    if asx_failures:
        st.warning(f"{len(asx_failures)} ASX ticker(s) failed and were skipped: {', '.join(asx_failures)}")

    with st.spinner("Aligning overnight drivers to ASX sessions (no look-ahead)..."):
        all_asx_dates = sorted(set().union(*[df.index for df in asx_outcomes.values() if not df.empty]))
        aligned = align_to_asx_sessions(driver_table, all_asx_dates)

    with st.spinner(f"Evaluating {len(HYPOTHESES)} hypotheses at day-level, stock-day, and per-stock granularity..."):
        results = [
            evaluate_hypothesis_detailed(h, aligned, asx_outcomes, ASX_THEME_STOCKS, COSTS)
            for h in HYPOTHESES
        ]

    st.divider()
    st.subheader("Summary — open→close, day-level, by hypothesis")
    st.caption("status and driver_sign_convention columns flag Coal (experimental) and REITs (inverted sign) — check those before reading their numbers the same way as everything else.")
    summary_rows = []
    for r in results:
        status, sign_note = tags_for(r["hypothesis_id"])
        if r.get("n_matched_days", 0) == 0:
            summary_rows.append({"id": r["hypothesis_id"], "status": status, "days": 0, "note": r.get("note")})
            continue
        val = r["day_level"]["open_to_close_pct"]["validation"]
        test = r["day_level"]["open_to_close_pct"]["test"]
        summary_rows.append({
            "id": r["hypothesis_id"], "direction": r["direction"], "theme": r["theme"],
            "status": status, "inverted_sign": bool(sign_note),
            "n_train": r["n_train_days"], "n_val": r["n_validation_days"], "n_test": r["n_test_days"],
            "val_band": val.get("band"), "val_win_rate": val.get("win_rate"), "val_expectancy_net": val.get("expectancy_net"),
            "test_band": test.get("band"), "test_win_rate": test.get("win_rate"), "test_expectancy_net": test.get("expectancy_net"),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Full detail per hypothesis")
    for r in results:
        status, sign_note = tags_for(r["hypothesis_id"])
        label = r.get("label", r["hypothesis_id"])
        badge = f" [EXPERIMENTAL]" if status == "experimental" else ""
        badge += f" [INVERTED SIGN]" if sign_note else ""
        with st.expander(f"{r['hypothesis_id']}{badge} — {label} (matched: {r.get('n_matched_days', 0)})"):
            if sign_note:
                st.warning(sign_note)
            if status == "experimental":
                st.warning("Marked EXPERIMENTAL: the driver-to-basket link for this theme is weaker than the "
                          "commodity-future-based themes. Do not give this equal weight just because the numbers look good.")
            if r.get("n_matched_days", 0) == 0:
                st.write(r.get("note", "No matches"))
                continue
            for col in OUTCOME_COLUMNS:
                st.markdown(f"**{col}**")
                split_df = pd.DataFrame(r["day_level"][col]).T
                st.dataframe(split_df, use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Day-level + stock-day CSV", data=flatten_results(results).to_csv(index=False),
                            file_name="v2_scanner_results.csv", mime="text/csv", use_container_width=True)
    with col2:
        st.download_button("Per-stock CSV", data=flatten_per_stock(results).to_csv(index=False),
                            file_name="v2_scanner_per_stock.csv", mime="text/csv", use_container_width=True)
    st.caption("Both CSVs carry status and driver_sign_convention columns on every row — filter or sort on those before drawing conclusions.")
else:
    st.info(f"Tap to run all {len(HYPOTHESES)} hypotheses. First run will take longer than Phase 1 — more tickers involved.")
