"""
V3 REVERSE DISCOVERY — Streamlit app (one button)
=====================================================
Run:  streamlit run reverse_discovery_app.py
Needs: v3_reverse_discovery.py + the existing repo files (intraday_engine.py V3,
historical_data.py, backtest.py, config_v1.py, eodhd_logic.py, intraday_data.py,
driver_threshold_sanity_check.py) and `pyarrow` in requirements.txt.
Secret: EODHD token in Streamlit secrets (any key containing 'EODHD').
"""
import io
import time
import zipfile

import pandas as pd
import streamlit as st

import v3_reverse_discovery as rd

st.set_page_config(page_title="V3 Reverse Discovery", layout="wide")
st.title("V3 Reverse Discovery — 16-stock pilot")
st.caption(
    "Which pre-open driver conditions separate morning-mover days from ordinary days? "
    "Found on TRAIN only → frozen → checked once on VALIDATION → once on TEST. "
    "Zero validated candidates is a legitimate result."
)

with st.expander("Locked rules (read-only)", expanded=False):
    st.markdown(
        f"""
- **Search:** 78 pre-registered driver conditions × UP/DOWN side, single driver only.
- **Targets:** +1/2/3/4% from the 10:00 open, LONG and SHORT.
- **Discovery checkpoints:** 10:15, 10:30, 11:00. (11:30, 12:00, full session = description only.)
- **TRAIN candidate:** n ≥ {rd.MIN_TRAIN_N} signal days **and** uplift ≥ +{rd.MIN_TRAIN_UPLIFT_PP:.0f}pp vs that stock's own baseline.
  One candidate (best uplift) per stock/direction/target/checkpoint, then frozen.
- **Validation:** same direction, positive, keeps ≥ {int(rd.VALIDATION_RETENTION*100)}% of TRAIN uplift. No candidate #2 on failure.
- **Test:** validation-passers only, once, never used to tune.
- **Split:** 60/20/20 by position, per stock per condition. Opening gap is description only.
        """
    )


def get_token():
    try:
        for k in st.secrets:
            if "EODHD" in str(k).upper():
                v = st.secrets[k]
                if isinstance(v, str):
                    return v
        for k in ("api_token", "API_TOKEN"):
            if k in st.secrets:
                return st.secrets[k]
    except Exception:
        pass
    return None


def make_zip(frames):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        names = {"combos": "rd_1_candidates_validation_test.csv",
                 "matrix": "rd_2_ungated_train_matrix.csv",
                 "continuation": "rd_3_continuation_mfe_mae.csv",
                 "gap": "rd_4_gap_buckets.csv",
                 "meta": "rd_5_ticker_history.csv"}
        for k, fn in names.items():
            if k in frames and not frames[k].empty:
                z.writestr(fn, frames[k].to_csv(index=False))
    return buf.getvalue()


token = get_token()
if not token:
    st.error("No EODHD token found in Streamlit secrets.")

run = st.button("▶ Run reverse discovery (16 stocks)", type="primary", disabled=not token)

if run:
    t0 = time.time()
    status = st.empty()
    bar = st.progress(0.0)
    conditions = rd.build_all_conditions()
    status.info(f"Fetching {len({c[0] for c in conditions})} drivers from Yahoo…")
    driver_table, failed = rd.build_drivers(conditions)
    if failed:
        # one retry for anything that failed (usually a transient rate limit)
        time.sleep(5)
        again, failed2 = rd.build_drivers([c for c in conditions if c[0] in {n for n, _ in failed}])
        if not again.empty:
            driver_table = driver_table.join(again, how="outer")
        failed = failed2
    if failed:
        st.warning("Drivers that could not be fetched (their conditions are excluded, not hidden): "
                   + ", ".join(f"{n} ({t})" for n, t in failed))
    getter = rd.make_eodhd_clean_days_getter(token)

    def progress(i, n, msg):
        bar.progress(min(1.0, i / n))
        status.info(f"[{i}/{n}] {msg}")

    out = rd.run_all(rd.PILOT_TICKERS, getter, driver_table, conditions, progress=progress)
    frames = rd.results_to_frames(out)
    st.session_state["rd_frames"] = frames
    st.session_state["rd_zip"] = make_zip(frames)
    st.session_state["rd_errors"] = out["errors"]
    st.session_state["rd_dropped"] = sorted(set(out["dropped_drivers"]) | {n for n, _ in failed})
    st.session_state["rd_secs"] = time.time() - t0
    status.success(f"Done in {(time.time() - t0) / 60:.1f} min")

# ---------------- results (rendered from session state so downloads don't wipe them) ----------------
frames = st.session_state.get("rd_frames")
if frames:
    combos = frames["combos"]
    st.subheader("Summary")
    n_total = len(combos)
    n_cand = int((combos["status"] != "NOT_A_CANDIDATE").sum())
    n_pass = int((combos["status"] == "VALIDATION_PASSED").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Combinations searched", n_total)
    c2.metric("TRAIN candidates", n_cand)
    c3.metric("Passed validation", n_pass)
    if n_pass and "test_same_direction_positive" in combos:
        c4.metric("…and positive on test", int(combos["test_same_direction_positive"].fillna(False).sum()))
    else:
        c4.metric("…and positive on test", 0)
    st.caption(
        "A TRAIN candidate is only 'best of 156 conditions' and is expected to appear by chance. "
        "Only validation and test survivors carry weight; n_train / n_validation / n_test are always shown."
    )

    if st.session_state.get("rd_errors"):
        st.warning("Tickers with problems (not silently dropped): " + "; ".join(f"{t}: {m}" for t, m in st.session_state["rd_errors"]))
    if st.session_state.get("rd_dropped"):
        st.warning("Drivers excluded this run: " + ", ".join(st.session_state["rd_dropped"]))

    show_cols = ["ticker", "direction", "target_pct", "checkpoint", "driver", "driver_threshold_pct", "driver_side",
                 "n_train", "n_validation", "n_test", "train_signal_prob", "train_baseline_prob", "train_uplift_pp",
                 "val_uplift_pp", "test_signal_prob", "test_baseline_prob", "test_uplift_pp",
                 "test_median_time_to_target_min", "test_median_mae_before_target", "score_test", "reason"]

    passed = combos[combos["status"] == "VALIDATION_PASSED"]
    st.subheader("Validation passers (ranked by frozen sweet-spot score, test probability)")
    if passed.empty:
        st.info("No candidate passed validation. That is a valid result.")
    else:
        st.dataframe(passed[[c for c in show_cols if c in passed.columns]], use_container_width=True, hide_index=True)

    failed_df = combos[combos["status"] == "FAILED_VALIDATION"]
    with st.expander(f"Candidates that failed validation ({len(failed_df)})"):
        st.dataframe(failed_df[[c for c in show_cols[:15] + ["val_retention_of_train", "reason"] if c in failed_df.columns]],
                     use_container_width=True, hide_index=True)
    with st.expander("Combinations with no TRAIN candidate (with best near-miss)"):
        nc = combos[combos["status"] == "NOT_A_CANDIDATE"]
        st.dataframe(nc.dropna(axis=1, how="all"), use_container_width=True, hide_index=True)
    with st.expander("History used per stock"):
        st.dataframe(frames["meta"], use_container_width=True, hide_index=True)

    st.subheader("Downloads")
    st.download_button("⬇ All CSVs (one ZIP)", st.session_state["rd_zip"], file_name="v3_reverse_discovery_results.zip",
                       mime="application/zip", type="primary")
    names = [("combos", "rd_1_candidates_validation_test.csv"), ("matrix", "rd_2_ungated_train_matrix.csv"),
             ("continuation", "rd_3_continuation_mfe_mae.csv"), ("gap", "rd_4_gap_buckets.csv")]
    cols = st.columns(len(names))
    for col, (k, fn) in zip(cols, names):
        if k in frames and not frames[k].empty:
            col.download_button(fn, frames[k].to_csv(index=False), file_name=fn, mime="text/csv", key=f"dl_{k}")
