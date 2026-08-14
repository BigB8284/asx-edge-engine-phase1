"""
PHASE B — Full universe intraday discovery research
=========================================================
Runs the SAME intraday engine proven in Phase A (intraday_engine.py,
intraday_stats.py, intraday_data.py — none of that logic is touched
here) across all 34 hypotheses in config_v2.py, covering all 11 themes
including ones that failed the daily open->close test in V1/V2. A
failed daily hypothesis does NOT disqualify a theme from this run.

Grade A/B/C classification uses the rule pre-specified in
phase_b_classification.py, written BEFORE this file was run against
real data. Nothing here optimises, searches, or adjusts that rule.

This is discovery research, not the live scanner. Frozen V1/V2 results
and Phase A are untouched by anything in this file.

MEMORY FIX (2026-08-14): originally fetched and held all 39 tickers'
full intraday history in memory simultaneously (in a `ticker_data`
dict AND duplicated again by Streamlit's cache_data copy-on-read),
which blew past Streamlit Community Cloud's memory limit mid-run.
Restructured to process one ticker at a time: fetch, score against
every hypothesis that uses it, append results, then let it fall out
of scope before moving to the next ticker. Scoring math is unchanged —
only the order of operations and what's held in memory at once. Cache
is also capped to 1 entry so old tickers' raw bars get evicted, not
just locally released.

DIAGNOSTIC LOGGING (2026-08-14): app still hit resource limits after
the above fix, with no traceback (OOM-kill signature) and no browser-
side evidence of where it died, since the crash wipes the progress bar
along with the rest of the page. Added a per-ticker memory checkpoint
that prints to the SERVER-SIDE log (survives independently of the
browser tab) so we can see exactly which ticker it dies on and how
memory climbs leading up to it.
"""

import streamlit as st
import pandas as pd
from collections import defaultdict
from datetime import datetime, timezone

from config_v2 import DRIVERS, ASX_THEME_STOCKS, HYPOTHESES
from historical_data import build_driver_table, align_to_asx_sessions, fetch_raw_history
from backtest import chronological_split
from intraday_data import fetch_full_intraday_history, build_clean_day_groups
from intraday_engine import compute_day_outcomes, THRESHOLDS, CHECKPOINTS, MFE_MAE_WINDOWS
from intraday_stats import aggregate_outcomes, compute_baseline_delta, format_summary_line
from phase_b_classification import (
    classify_finding, CLASSIFICATION_ANCHOR_THRESHOLD_PCT, CLASSIFICATION_ANCHOR_CHECKPOINT,
)
import resource
import gc

st.set_page_config(page_title="Phase B — Full Universe Discovery", layout="wide")
st.title("Phase B — Full Universe Intraday Discovery")
st.caption("All 34 hypotheses, all 11 themes. Discovery research using the Phase A-validated engine, unchanged. Not the live scanner.")

by_id = {h["id"]: h for h in HYPOTHESES}
INTRADAY_START = datetime(2020, 10, 12, tzinfo=timezone.utc)

try:
    EODHD_TOKEN = st.secrets["EODHD_API_TOKEN"]
except Exception:
    st.error("No EODHD_API_TOKEN found in Streamlit secrets.")
    st.stop()


def to_eodhd_ticker(yahoo_ticker):
    if yahoo_ticker.endswith(".AX"):
        return yahoo_ticker[:-3] + ".AU"
    return yahoo_ticker


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def cached_fetch_daily(ticker):
    return fetch_raw_history(ticker)


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_driver_table():
    return build_driver_table(list(DRIVERS.keys()), fetch_fn=cached_fetch_daily, drivers_lookup=DRIVERS)


# max_entries=1: only the ticker currently being processed needs to stay
# cached. Without this cap, Streamlit's cache keeps every ticker's raw
# intraday history resident for the whole run on top of the working copy —
# that's what caused the original OOM kill.
@st.cache_data(ttl=60 * 60 * 12, show_spinner=False, max_entries=1)
def load_intraday_for_ticker(ticker):
    eodhd_ticker = to_eodhd_ticker(ticker)
    end = datetime.now(timezone.utc)
    raw_bars, failed_windows = fetch_full_intraday_history(eodhd_ticker, EODHD_TOKEN, INTRADAY_START, end)
    clean_days, excluded_days, flagged_moves, outside_info = build_clean_day_groups(raw_bars)
    return clean_days, excluded_days, flagged_moves, outside_info, failed_windows


def all_needed_tickers():
    """Union of every ticker across EVERY theme referenced by any of
    the 34 hypotheses — the full universe, not just Grade-A."""
    themes = {h["theme"] for h in HYPOTHESES}
    tickers = set()
    for t in themes:
        tickers.update(ASX_THEME_STOCKS[t])
    return sorted(tickers)


def outcomes_for_dates(clean_days, date_strs, direction):
    outcomes = []
    for ds in date_strs:
        if ds in clean_days:
            o = compute_day_outcomes(clean_days[ds], direction)
            if o is not None:
                outcomes.append(o)
    return outcomes


def get_anchor_stats(agg):
    """Pulls the pre-specified classification anchor (2%, full_session)
    absolute probability out of an aggregated stats dict."""
    return agg["thresholds"].get(CLASSIFICATION_ANCHOR_THRESHOLD_PCT)


def build_ticker_to_themes():
    m = defaultdict(set)
    for theme, theme_tickers in ASX_THEME_STOCKS.items():
        for t in theme_tickers:
            m[t].add(theme)
    return m


def build_theme_to_hypotheses():
    m = defaultdict(list)
    for h in HYPOTHESES:
        m[h["theme"]].append(h)
    return m


if st.button("Run Phase B (full universe)", type="primary", use_container_width=True):
    with st.spinner("Pulling driver history..."):
        driver_table, driver_failures = load_driver_table()
    if driver_failures:
        st.warning(f"{len(driver_failures)} driver ticker(s) failed: {driver_failures}")

    tickers = all_needed_tickers()
    st.info(f"Full universe: {len(tickers)} tickers across 11 themes: {tickers}")

    ticker_to_themes = build_ticker_to_themes()
    theme_to_hypotheses = build_theme_to_hypotheses()

    all_summary_rows = []
    full_detail_rows = []
    fail_summary = []
    any_clean_days_found = False

    print(f"=== Phase B run starting: {len(tickers)} tickers ===", flush=True)

    progress = st.progress(0, text="Processing tickers...")
    for i, ticker in enumerate(tickers):
        clean_days, excluded_days, flagged_moves, outside_info, failed_windows = load_intraday_for_ticker(ticker)

        if failed_windows:
            fail_summary.append((ticker, failed_windows))

        if not clean_days:
            progress.progress((i + 1) / len(tickers), text=f"{ticker}: no clean days ({i + 1}/{len(tickers)})")
            gc.collect()
            mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f"[{i + 1}/{len(tickers)}] {ticker}: no clean days, peak RSS so far: {mem_mb:.0f} MB", flush=True)
            continue

        any_clean_days_found = True

        # Align drivers to THIS ticker's own trading dates only — equivalent
        # to the old global-union approach, since outcomes_for_dates() was
        # always filtering down to per-ticker dates anyway.
        ticker_dates = sorted(pd.Timestamp(d) for d in clean_days.keys())
        aligned = align_to_asx_sessions(driver_table, ticker_dates)

        relevant_themes = ticker_to_themes.get(ticker, set())
        relevant_hypotheses = [h for theme in relevant_themes for h in theme_to_hypotheses.get(theme, [])]

        baseline_cache_for_ticker = {}  # direction -> aggregated baseline, reused across hypotheses sharing a direction

        for h in relevant_hypotheses:
            hid = h["id"]
            direction = h["direction"]
            theme = h["theme"]
            status = h.get("status", "")
            sign_note = h.get("driver_sign_convention", "")
            usable_from = pd.Timestamp(h["usable_from"])

            driver_slice = aligned[aligned.index >= usable_from]
            matched_dates = [d for d in driver_slice.index if h["condition"](driver_slice.loc[d])]
            train_d, val_d, test_d = chronological_split(matched_dates)
            split_dates = {"train": train_d, "validation": val_d, "test": test_d}

            if direction not in baseline_cache_for_ticker:
                outcomes = [compute_day_outcomes(bars, direction) for bars in clean_days.values()]
                baseline_cache_for_ticker[direction] = aggregate_outcomes([o for o in outcomes if o is not None])
            baseline_agg = baseline_cache_for_ticker[direction]

            split_aggs = {}
            for split_name, dates in split_dates.items():
                date_strs = [str(d.date()) for d in dates]
                outcomes = outcomes_for_dates(clean_days, date_strs, direction)
                split_aggs[split_name] = aggregate_outcomes(outcomes) if outcomes else None

            val_agg, test_agg = split_aggs.get("validation"), split_aggs.get("test")
            val_anchor = get_anchor_stats(val_agg) if val_agg and val_agg["n_days"] > 0 else None
            test_anchor = get_anchor_stats(test_agg) if test_agg and test_agg["n_days"] > 0 else None
            baseline_anchor = get_anchor_stats(baseline_agg)

            val_delta = (val_anchor["probability"] - baseline_anchor["probability"]) if val_anchor and baseline_anchor else None
            test_delta = (test_anchor["probability"] - baseline_anchor["probability"]) if test_anchor and baseline_anchor else None
            n_val = val_anchor["n_days"] if val_anchor else None
            n_test = test_anchor["n_days"] if test_anchor else None

            grade = classify_finding(val_delta, test_delta, n_val, n_test)

            all_summary_rows.append({
                "hypothesis_id": hid, "theme": theme, "direction": direction, "ticker": ticker,
                "status": status, "inverted_sign": bool(sign_note), "grade": grade,
                "val_signal_prob_2pct": val_anchor["probability"] if val_anchor else None,
                "val_n": n_val,
                "test_signal_prob_2pct": test_anchor["probability"] if test_anchor else None,
                "test_n": n_test,
                "baseline_prob_2pct": baseline_anchor["probability"] if baseline_anchor else None,
                "baseline_n": baseline_agg["n_days"],
                "val_delta_pp": round(val_delta, 1) if val_delta is not None else None,
                "test_delta_pp": round(test_delta, 1) if test_delta is not None else None,
                "stability_min_delta_pp": round(min(val_delta, test_delta), 1) if (val_delta is not None and test_delta is not None) else None,
            })

            for split_name, agg in split_aggs.items():
                if agg is None or agg["n_days"] == 0:
                    continue
                delta = compute_baseline_delta(agg, baseline_agg)
                for t in THRESHOLDS:
                    for cp in CHECKPOINTS:
                        d = delta["checkpoints"][cp][t]
                        full_detail_rows.append({
                            "hypothesis_id": hid, "theme": theme, "direction": direction, "ticker": ticker,
                            "status": status, "inverted_sign": bool(sign_note), "grade": grade,
                            "split": split_name, "threshold_pct": t, "checkpoint": cp,
                            "signal_probability": d["signal_probability"], "signal_n": d["signal_n"],
                            "baseline_probability": d["baseline_probability"], "baseline_n": d["baseline_n"],
                            "delta_pp": d["delta_pp"],
                            "median_time_to_threshold_min": agg["thresholds"][t]["median_time_to_threshold_min"],
                            "median_mae_before_reached": agg["thresholds"][t]["median_mae_before_reached"],
                        })
                for w in MFE_MAE_WINDOWS:
                    full_detail_rows.append({
                        "hypothesis_id": hid, "theme": theme, "direction": direction, "ticker": ticker,
                        "status": status, "inverted_sign": bool(sign_note), "grade": grade,
                        "split": split_name, "window": str(w), "metric": "MFE_MAE",
                        "median_mfe": agg["windows"][w]["median_mfe"], "median_mae": agg["windows"][w]["median_mae"],
                        "median_time_to_mfe_min": agg["windows"][w]["median_time_to_mfe_min"],
                    })

        progress.progress(
            (i + 1) / len(tickers),
            text=f"{ticker}: {len(clean_days)} clean days, {len(relevant_hypotheses)} hypotheses scored ({i + 1}/{len(tickers)})",
        )
        gc.collect()
        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"[{i + 1}/{len(tickers)}] {ticker}: {len(clean_days)} clean days, "
              f"{len(relevant_hypotheses)} hypotheses, summary_rows={len(all_summary_rows)}, "
              f"detail_rows={len(full_detail_rows)}, peak RSS so far: {mem_mb:.0f} MB", flush=True)
        # clean_days / aligned / baseline_cache_for_ticker fall out of scope
        # on the next loop iteration — nothing carried over between tickers.

    progress.empty()

    if fail_summary:
        st.warning(f"{len(fail_summary)} ticker(s) had at least one failed fetch window: {[t for t, _ in fail_summary]}")

    if not any_clean_days_found:
        st.error("No clean intraday days found anywhere — stopping.")
        st.stop()

    summary_df = pd.DataFrame(all_summary_rows)
    grade_order = {"A": 0, "B": 1, "C": 2}
    summary_df["_sort"] = summary_df["grade"].map(grade_order)
    summary_df = summary_df.sort_values(["_sort", "stability_min_delta_pp"], ascending=[True, False]).drop(columns="_sort")

    st.divider()
    st.subheader(f"Ranked summary — all {len(summary_df)} (hypothesis, ticker) pairs")
    st.caption(f"Grading anchor: +{CLASSIFICATION_ANCHOR_THRESHOLD_PCT:.0f}% threshold, {CLASSIFICATION_ANCHOR_CHECKPOINT}. Ranked by grade, then by worst-of-(validation,test) delta — stability first, not a spectacular single split.")

    for grade in ["A", "B", "C"]:
        subset = summary_df[summary_df["grade"] == grade]
        st.markdown(f"### Grade {grade} — {len(subset)} pairs")
        if subset.empty:
            st.write("None.")
            continue
        display_cols = ["hypothesis_id", "theme", "direction", "ticker", "status", "inverted_sign",
                        "val_signal_prob_2pct", "val_n", "test_signal_prob_2pct", "test_n",
                        "baseline_prob_2pct", "val_delta_pp", "test_delta_pp"]
        st.dataframe(subset[display_cols], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Downloads")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Ranked summary CSV (A/B/C, +2% anchor)", data=summary_df.to_csv(index=False),
                            file_name="phase_b_summary.csv", mime="text/csv", use_container_width=True)
    with c2:
        st.download_button("Full detail CSV (all thresholds/checkpoints/windows)", data=pd.DataFrame(full_detail_rows).to_csv(index=False),
                            file_name="phase_b_full_detail.csv", mime="text/csv", use_container_width=True)
    st.caption("Full detail CSV has every 1/2/3/5% threshold x every checkpoint x every MFE/MAE window, per split — the +2%/full_session anchor above is only the grading cut, not the whole picture.")
else:
    st.info(f"Tap to run. {len(HYPOTHESES)} hypotheses across 11 themes — this pulls intraday history for the full ticker universe and will take considerably longer than Phase A's 8-ticker run.")
