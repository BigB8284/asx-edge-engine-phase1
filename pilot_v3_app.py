"""
V3 PILOT — 16-stock proof-of-concept for the new morning-excursion scanner
================================================================================
Purpose (per approved spec): prove the new V3 architecture and
methodology work correctly, on a small representative mix, BEFORE
downloading/testing the full ~64-stock universe. Not the live scanner.

Reuses, UNCHANGED:
  - intraday_data.py's fetch_full_intraday_history / build_clean_day_groups
  - historical_data.py's build_driver_table / align_to_asx_sessions / fetch_raw_history
  - backtest.py's chronological_split
  - intraday_engine.py's compute_day_outcomes (V3-edited: THRESHOLDS is now
    [1,2,3,4], CHECKPOINTS gains 11:30 — see that file's own changelog note)
  - intraday_stats.py's aggregate_outcomes / compute_baseline_delta
    (V3-added: compute_target_before_adverse, a pure aggregation needing
    no engine changes)

New for V3:
  - v3_classification.py — the FROZEN sweet-spot score, eligibility gate,
    and VALIDATED/WATCH/EXPERIMENTAL/NO EDGE confidence labels
  - v3_persistence.py — raw 5-minute bars saved once, never re-fetched
  - v3_pilot_config.py — hypothesis coverage for the 16 pilot tickers,
    including 4 new single-ticker ADR/dual-listing hypotheses

Train -> LOCK -> validation -> test is enforced structurally: the sweet
spot is selected using ONLY train-split aggregates, then that exact
(threshold, checkpoint) combo is evaluated once on validation and once
on test. There is no code path that re-selects after seeing either.

Output ranks (ticker, hypothesis, direction) cards by sweet-spot score
(grouped visually by ticker), each showing its FULL ladder — every
scanner-eligible combo, gate-passing or not — with the sweet spot
starred. Nothing is hidden by the score; the score only picks the star.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timezone
import resource
import gc

from v3_pilot_config import PILOT_TICKERS, HYPOTHESES, ASX_THEME_STOCKS, DRIVERS, hypotheses_for_ticker
from historical_data import build_driver_table, align_to_asx_sessions, fetch_raw_history
from backtest import chronological_split
from intraday_data import fetch_full_intraday_history, build_clean_day_groups
from intraday_engine import compute_day_outcomes, THRESHOLDS, CHECKPOINTS
from intraday_stats import aggregate_outcomes, compute_target_before_adverse, DEFAULT_ADVERSE_LEVELS_PCT
import v3_classification as v3c
import v3_persistence as v3p

st.set_page_config(page_title="V3 Pilot — Morning Opportunity Scanner", layout="wide")
st.title("V3 Pilot — Morning Opportunity Scanner")
st.caption("16 tickers, proving the new architecture before the full ~64-stock universe. Not the live scanner. "
           "Train -> lock -> validation -> test is structurally enforced, not just followed by convention.")

INTRADAY_START = datetime(2020, 10, 12, tzinfo=timezone.utc)
BATCH_SIZE = 8
SCANNER_CHECKPOINTS = [cp for cp in CHECKPOINTS if cp in v3c.SPEED_BONUS]  # excludes full_session

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


def fetch_intraday_for_ticker(ticker):
    """The raw EODHD call — only invoked by v3_persistence.get_or_fetch
    on a genuine cache miss."""
    eodhd_ticker = to_eodhd_ticker(ticker)
    end = datetime.now(timezone.utc)
    raw_bars, failed_windows = fetch_full_intraday_history(eodhd_ticker, EODHD_TOKEN, INTRADAY_START, end)
    clean_days, excluded_days, flagged_moves, outside_info = build_clean_day_groups(raw_bars)
    return clean_days, excluded_days, flagged_moves, outside_info, failed_windows


def outcomes_for_dates(clean_days, date_strs, direction):
    outcomes = []
    for ds in date_strs:
        if ds in clean_days:
            o = compute_day_outcomes(clean_days[ds], direction)
            if o is not None:
                outcomes.append(o)
    return outcomes


def build_ladder_and_select(train_agg, baseline_agg):
    """For every scanner-eligible (threshold, checkpoint) combo: pull
    train probability + baseline + delta, check the eligibility gate,
    score gate-passers, and return (full_ladder_rows, sweet_spot_or_None).
    full_ladder_rows includes EVERY combo, gated or not — nothing is
    dropped from the ladder itself, only from sweet-spot eligibility."""
    ladder_rows = []
    candidates = []

    for t in THRESHOLDS:
        mae_before = train_agg["thresholds"][t]["median_mae_before_reached"]
        for cp in SCANNER_CHECKPOINTS:
            sig = train_agg["checkpoints"][cp][t]
            base = baseline_agg["checkpoints"][cp][t]
            delta = round(sig["probability"] - base["probability"], 1)
            gate_passed = v3c.passes_eligibility_gate(delta, sig["n_days"], mae_before, t)
            score = v3c.score_opportunity(sig["probability"], t, cp) if gate_passed else None

            row = {
                "threshold_pct": t, "checkpoint": cp,
                "train_probability": sig["probability"], "train_n": sig["n_days"],
                "baseline_probability": base["probability"], "delta_pp": delta,
                "median_mae_before_reached": mae_before,
                "gate_passed": gate_passed, "score": score,
            }
            ladder_rows.append(row)
            if gate_passed:
                candidates.append({"threshold_pct": t, "checkpoint": cp,
                                    "train_probability_pct": sig["probability"], "score": score})

    sweet_spot = v3c.select_sweet_spot(candidates)
    return ladder_rows, sweet_spot


def evaluate_locked_combo(clean_days, direction, date_strs, threshold, checkpoint, baseline_agg):
    """Evaluates ONE already-locked (threshold, checkpoint) combo against
    a single split's dates. Never searches — just measures."""
    outcomes = outcomes_for_dates(clean_days, date_strs, direction)
    if not outcomes:
        return None
    agg = aggregate_outcomes(outcomes)
    sig = agg["checkpoints"][checkpoint][threshold]
    base = baseline_agg["checkpoints"][checkpoint][threshold]
    return {
        "probability": sig["probability"], "n": sig["n_days"],
        "delta_pp": round(sig["probability"] - base["probability"], 1),
        "target_before_adverse": compute_target_before_adverse(outcomes)[threshold],
    }


def process_ticker(ticker, driver_table, mem_log):
    """Fetches (or loads from persistence) this ticker's data, then
    builds every (hypothesis, direction) card for it. Returns a list of
    card dicts, or a list containing a single error-flag dict if the
    fetch produced no usable data."""
    clean_days, exclusions, from_cache = v3p.get_or_fetch(ticker, lambda: fetch_intraday_for_ticker(ticker))

    gc.collect()
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    mem_log.append(f"{ticker}: {'cache hit' if from_cache else 'fetched fresh'}, "
                    f"{len(clean_days) if clean_days else 0} clean days, peak RSS {mem_mb:.0f} MB")

    if not clean_days:
        return [{"ticker": ticker, "error": "No clean intraday days available at all — data/code problem, not a null result."}]

    ticker_dates = sorted(pd.Timestamp(d) for d in clean_days.keys())
    aligned = align_to_asx_sessions(driver_table, ticker_dates)

    cards = []
    baseline_cache = {}  # direction -> baseline agg, reused across this ticker's hypotheses sharing a direction

    for h in hypotheses_for_ticker(ticker):
        direction = h["direction"]
        usable_from = pd.Timestamp(h["usable_from"])

        driver_slice = aligned[aligned.index >= usable_from]
        matched_dates = [d for d in driver_slice.index if h["condition"](driver_slice.loc[d])]
        train_d, val_d, test_d = chronological_split(matched_dates)

        if direction not in baseline_cache:
            all_outcomes = [compute_day_outcomes(bars, direction) for bars in clean_days.values()]
            baseline_cache[direction] = aggregate_outcomes([o for o in all_outcomes if o is not None])
        baseline_agg = baseline_cache[direction]

        train_outcomes = outcomes_for_dates(clean_days, [str(d.date()) for d in train_d], direction)
        if not train_outcomes:
            cards.append({
                "ticker": ticker, "hypothesis_id": h["id"], "theme": h["theme"], "direction": direction,
                "error": f"No matched training days for this hypothesis (usable_from={h['usable_from']}) "
                         f"— possibly too little history yet, not a tested-and-failed result.",
                "driver_sign_convention": h.get("driver_sign_convention"), "status": h.get("status"),
            })
            continue

        train_agg = aggregate_outcomes(train_outcomes)
        ladder_rows, sweet_spot = build_ladder_and_select(train_agg, baseline_agg)

        # BUGFIX (found via real pilot data, 2026-08-16): n_train/n_validation/n_test
        # must be the ACTUAL count of days that produced a usable outcome (what the
        # gate and confidence classifier really use), not the raw count of matched
        # dates from the driver condition. Those two numbers can diverge when a
        # matched date falls on a day with no usable intraday data — the classifier
        # was already correctly using the real count, but the CSV was reporting a
        # different, possibly larger number, making the displayed n inconsistent
        # with the confidence label next to it. Both counts are now reported,
        # clearly labeled, so any gap between them is visible rather than hidden.
        n_train_matched_dates = len(train_d)
        n_validation_matched_dates = len(val_d)
        n_test_matched_dates = len(test_d)
        n_train_actual = train_agg["n_days"]

        card = {
            "ticker": ticker, "hypothesis_id": h["id"], "theme": h["theme"], "direction": direction,
            "driver_sign_convention": h.get("driver_sign_convention"), "status": h.get("status"),
            "n_train": n_train_actual, "n_train_matched_dates": n_train_matched_dates,
            "n_validation_matched_dates": n_validation_matched_dates, "n_test_matched_dates": n_test_matched_dates,
            "ladder": ladder_rows, "sweet_spot": sweet_spot,
        }

        if sweet_spot is None:
            card["confidence"] = "EXPERIMENTAL"
            card["confidence_note"] = "No combo cleared the eligibility gate on training data for this hypothesis/ticker."
            card["n_validation"] = n_validation_matched_dates  # no locked combo evaluated, matched-dates is all we have
            card["n_test"] = n_test_matched_dates
        else:
            t_locked, cp_locked = sweet_spot["threshold_pct"], sweet_spot["checkpoint"]
            val_result = evaluate_locked_combo(clean_days, direction, [str(d.date()) for d in val_d], t_locked, cp_locked, baseline_agg)
            test_result = evaluate_locked_combo(clean_days, direction, [str(d.date()) for d in test_d], t_locked, cp_locked, baseline_agg)
            card["locked_threshold"] = t_locked
            card["locked_checkpoint"] = cp_locked
            card["validation_result"] = val_result
            card["test_result"] = test_result
            # The REAL n used for classification — matches what classify_confidence sees.
            card["n_validation"] = val_result["n"] if val_result else 0
            card["n_test"] = test_result["n"] if test_result else 0
            if card["n_validation"] != n_validation_matched_dates or card["n_test"] != n_test_matched_dates:
                card["n_gap_note"] = (f"{n_validation_matched_dates - card['n_validation']} matched validation date(s) and "
                                      f"{n_test_matched_dates - card['n_test']} matched test date(s) had no usable intraday "
                                      f"outcome — real n is lower than the raw matched-date count.")
            card["confidence"] = v3c.classify_confidence(
                val_result["delta_pp"] if val_result else None,
                test_result["delta_pp"] if test_result else None,
                val_result["n"] if val_result else None,
                test_result["n"] if test_result else None,
            )

        cards.append(card)

    return cards


def compute_pilot_batch(batch_tickers, batch_num, total_batches):
    with st.spinner("Pulling driver history..."):
        driver_table, driver_failures = load_driver_table()

    mem_log = []
    all_cards = []
    progress = st.progress(0, text="Processing tickers...")
    for i, ticker in enumerate(batch_tickers):
        cards = process_ticker(ticker, driver_table, mem_log)
        all_cards.extend(cards)
        progress.progress((i + 1) / len(batch_tickers), text=f"{ticker} ({i + 1}/{len(batch_tickers)})")
    progress.empty()

    return {"cards": all_cards, "driver_failures": driver_failures, "mem_log": mem_log,
            "batch_tickers": batch_tickers}


def format_pct(p):
    return f"{p:.0f}%" if p is not None else "n/a"


def render_card(card, rank=None):
    if "error" in card and "ladder" not in card:
        st.error(f"**{card['ticker']}**" + (f" / {card.get('hypothesis_id','')}" if card.get('hypothesis_id') else "")
                  + f" — {card['error']}")
        return

    header = f"#{rank} " if rank is not None else ""
    header += f"{card['ticker']} — {card['direction']} ({card['theme']}, {card['hypothesis_id']})"
    st.markdown(f"### {header}")

    if card.get("driver_sign_convention"):
        st.warning(f"⚠️ INVERTED SIGN CONVENTION: {card['driver_sign_convention']}")
    if card.get("status") == "experimental":
        st.info("Flagged EXPERIMENTAL driver relationship (weaker/more diffuse than a commodity-future-based theme) — "
                "treat any result here with extra caution regardless of confidence label.")

    st.markdown(f"**Confidence: {card['confidence']}**" + (f" — {card.get('confidence_note','')}" if card.get("confidence_note") else ""))
    st.caption(f"n train/validation/test: {card['n_train']}/{card['n_validation']}/{card['n_test']}")
    if card.get("n_gap_note"):
        st.caption(f"⚠️ {card['n_gap_note']}")

    # Full ladder — every combo, gated or not
    ladder_df = pd.DataFrame(card["ladder"])
    ladder_df["label"] = ladder_df.apply(lambda r: f"+{r['threshold_pct']:.0f}% by {r['checkpoint']}", axis=1)
    is_sweet_spot = lambda r: (card["sweet_spot"] is not None
                                and r["threshold_pct"] == card["sweet_spot"]["threshold_pct"]
                                and r["checkpoint"] == card["sweet_spot"]["checkpoint"])
    ladder_df["★"] = ladder_df.apply(lambda r: "★ SWEET SPOT" if is_sweet_spot(r) else "", axis=1)
    ladder_df["gate"] = ladder_df["gate_passed"].map({True: "eligible", False: "gated out"})
    display_cols = ["label", "train_probability", "baseline_probability", "delta_pp", "train_n", "gate", "score", "★"]
    st.dataframe(ladder_df[display_cols].rename(columns={
        "label": "Setup", "train_probability": "Train prob %", "baseline_probability": "Baseline %",
        "delta_pp": "Δpp", "train_n": "Train n", "score": "Opportunity score",
    }), use_container_width=True, hide_index=True)
    st.caption("Opportunity score ≠ probability — it only orders this ladder to pick the ★, shown separately from every probability above.")

    if card.get("sweet_spot") is not None:
        vr, tr = card.get("validation_result"), card.get("test_result")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Validation (locked combo, evaluated once)**")
            if vr:
                st.write(f"Probability: {format_pct(vr['probability'])} | n={vr['n']} | Δ={vr['delta_pp']}pp")
                st.write("Target-before-adverse:", {f"before {a}%": format_pct(s["probability"]) for a, s in vr["target_before_adverse"].items()})
            else:
                st.write("No matched validation days.")
        with c2:
            st.markdown("**Test (locked combo, evaluated once)**")
            if tr:
                st.write(f"Probability: {format_pct(tr['probability'])} | n={tr['n']} | Δ={tr['delta_pp']}pp")
                st.write("Target-before-adverse:", {f"before {a}%": format_pct(s["probability"]) for a, s in tr["target_before_adverse"].items()})
            else:
                st.write("No matched test days.")
    st.divider()


def render_pilot_result(result, batch_num, total_batches):
    if result["driver_failures"]:
        st.warning(f"{len(result['driver_failures'])} driver ticker(s) failed: {result['driver_failures']}")

    with st.expander("Memory / fetch log for this batch"):
        for line in result["mem_log"]:
            st.text(line)

    cards = result["cards"]
    scoreable = [c for c in cards if c.get("sweet_spot") is not None]
    unscoreable = [c for c in cards if c.get("sweet_spot") is None]

    scoreable_sorted = sorted(scoreable, key=lambda c: c["sweet_spot"]["score"], reverse=True)

    st.subheader(f"Batch {batch_num}/{total_batches} — ranked cards with a sweet spot ({len(scoreable_sorted)})")
    for i, card in enumerate(scoreable_sorted, start=1):
        render_card(card, rank=i)

    if unscoreable:
        st.subheader(f"No sweet spot cleared the gate ({len(unscoreable)}) — shown, not discarded")
        for card in unscoreable:
            render_card(card)

    st.divider()
    st.subheader("Downloads")
    summary_rows = []
    for c in cards:
        if "ladder" not in c:
            continue
        row = {"ticker": c["ticker"], "hypothesis_id": c["hypothesis_id"], "theme": c["theme"],
               "direction": c["direction"], "confidence": c["confidence"],
               "n_train": c["n_train"], "n_validation": c["n_validation"], "n_test": c["n_test"],
               "n_train_matched_dates": c.get("n_train_matched_dates"),
               "n_validation_matched_dates": c.get("n_validation_matched_dates"),
               "n_test_matched_dates": c.get("n_test_matched_dates"),
               "n_gap_note": c.get("n_gap_note", "")}
        if c.get("sweet_spot"):
            row["sweet_spot_threshold"] = c["sweet_spot"]["threshold_pct"]
            row["sweet_spot_checkpoint"] = c["sweet_spot"]["checkpoint"]
            row["sweet_spot_score"] = round(c["sweet_spot"]["score"], 2)
            row["train_probability"] = c["sweet_spot"]["train_probability_pct"]
        if c.get("validation_result"):
            row["validation_probability"] = c["validation_result"]["probability"]
            row["validation_delta_pp"] = c["validation_result"]["delta_pp"]
        if c.get("test_result"):
            row["test_probability"] = c["test_result"]["probability"]
            row["test_delta_pp"] = c["test_result"]["delta_pp"]
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    st.download_button(f"Batch {batch_num} pilot report CSV", data=summary_df.to_csv(index=False),
                        file_name=f"v3_pilot_report_batch{batch_num}.csv", mime="text/csv",
                        key=f"pilot_dl_{batch_num}")


# ---- Batch layout ----
batches = [PILOT_TICKERS[i:i + BATCH_SIZE] for i in range(0, len(PILOT_TICKERS), BATCH_SIZE)]

st.divider()
st.subheader(f"Run each batch ({len(batches)} batches, {len(PILOT_TICKERS)} tickers total)")

storage = v3p.storage_summary()
st.caption(f"Persisted data on disk right now: {storage['n_tickers_cached']} ticker(s), "
           f"{storage['total_size_mb']} MB. Already-cached tickers skip EODHD entirely.")

for batch_num, batch_tickers in enumerate(batches, start=1):
    state_key = f"pilot_batch_{batch_num}_result"
    with st.expander(f"Batch {batch_num} of {len(batches)} — {', '.join(batch_tickers)}"):
        if st.button(f"Run Batch {batch_num}", type="primary", use_container_width=True, key=f"run_pilot_{batch_num}"):
            st.session_state[state_key] = compute_pilot_batch(batch_tickers, batch_num, len(batches))
        if state_key in st.session_state:
            render_pilot_result(st.session_state[state_key], batch_num, len(batches))

st.divider()
if st.button("Export all persisted raw data as ZIP (for external archival)"):
    zip_path = v3p.export_all_as_zip()
    if zip_path:
        with open(zip_path, "rb") as f:
            st.download_button("Download raw data export", data=f.read(), file_name="v3_raw_data_export.zip", mime="application/zip")
    else:
        st.info("Nothing persisted yet — run a batch first.")
