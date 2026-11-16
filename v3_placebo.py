"""
V3 PLACEBO CALIBRATION — Streamlit app (separate from the reverse-discovery app)
=====================================================================================
Run:  streamlit run placebo_app.py
Needs: v3_placebo.py + v3_reverse_discovery.py (unchanged) + the usual repo files, and
`pyarrow` in requirements.txt. Reads the EODHD token from Streamlit secrets (any key containing 'EODHD').

Tab 1: placebo calibration (real run reproduced + N shifted-driver repetitions).
Tab 2: timestamp audits (AUD/USD daily-bar timing; EODHD 5-minute bar labelling) - fast, run it first.
Nothing here changes the original app, code, thresholds or results.
"""
import time

import pandas as pd
import streamlit as st

import v3_reverse_discovery as rd
import v3_placebo as pl

OUT_DIR = "v3_placebo_out"

st.set_page_config(page_title="V3 Placebo Calibration", layout="wide")
st.title("V3 Placebo Calibration")
st.caption(
    "Are the reverse-discovery findings stronger than what randomly re-dated overnight data produces "
    "through the identical process? The original app, code and CSVs are untouched."
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


token = get_token()
if not token:
    st.error("No EODHD token found in Streamlit secrets.")

tab1, tab2, tab3 = st.tabs(["1 · Placebo calibration", "2 · Timestamp audits", "How it works"])

# ============================================================================ TAB 1
with tab1:
    st.markdown(
        "Runs the **real data again** (should reproduce 152 / 67 / 34) and then **N placebo repetitions** in which every "
        "stock's driver block is shifted as one piece by at least six months. Each repetition re-runs the complete "
        "discovery → validation → test process over all 156 conditions."
    )
    c1, c2, c3 = st.columns(3)
    n_reps = c1.selectbox("Placebo repetitions", [20, 50, 100], index=0,
                          help="Reps are cheap once data is loaded; 100 adds under a minute. 20 = first 20 of the 100.")
    seed = c2.number_input("Random seed", value=pl.DEFAULT_SEED, step=1)
    conv_label = c3.selectbox("Checkpoint rule", ["original (as run)", "strict (sensitivity only)"], index=0,
                              help="'original' is exactly the first experiment. 'strict' only counts bars that have "
                                   "finished by the checkpoint; use it only to see how much that rule matters.")
    convention = "original" if conv_label.startswith("original") else "strict"
    if convention == "strict":
        st.warning("Strict rule selected: this is a sensitivity run. Real and placebo both use it, so they are still "
                   "comparable with each other, but not with the original 152 / 67 / 34.")

    run = st.button("▶ Run placebo calibration", type="primary", disabled=not token)

    if run:
        t0 = time.time()
        status, bar = st.empty(), st.progress(0.0)
        conditions = rd.build_all_conditions()
        status.info("Loading the 34 drivers from Yahoo (cached for 20 hours once complete)…")
        driver_table, failed = pl.load_or_build_drivers(conditions, OUT_DIR)
        if failed:
            st.warning("Drivers that could not be fetched: " + ", ".join(f"{n} ({t})" for n, t in failed)
                       + ". Their conditions would be missing, so the run is stopped. Press Run again in a minute "
                         "(this is almost always a temporary Yahoo limit).")
            st.stop()
        getter = rd.make_eodhd_clean_days_getter(token)

        def progress(stage, i, n, msg):
            frac = (i / max(1, n)) * (0.75 if stage == "data" else 0.25) + (0.0 if stage == "data" else 0.75)
            bar.progress(min(1.0, frac))
            status.info(msg)

        res = pl.run_calibration(rd.PILOT_TICKERS, getter, driver_table, conditions, n_reps=int(n_reps), seed=int(seed),
                                 convention=convention, out_dir=OUT_DIR, progress=progress)
        st.session_state["pl_res"] = res
        st.session_state["pl_zip"] = pl.build_results_zip(res)
        bar.progress(1.0)
        status.success(f"Done in {(time.time() - t0) / 60:.1f} min")

    res = st.session_state.get("pl_res")
    if res:
        rm, ref = res["real_metrics"], res["reference"]
        st.subheader("1. Did the app reproduce the original run?")
        if res["convention"] != "original":
            st.info("Strict-rule sensitivity run: no comparison with the original numbers is expected.")
        elif res["reproduces_reference"]:
            st.success(f"Yes. Real data through this app gives {rm['n_candidates']} candidates / {rm['n_val_pass']} validation "
                       f"passes / {rm['n_test_kept']} Test-kept, identical to the original run.")
        else:
            st.error("Not identical to the original run. " + ", ".join(
                f"{k}: original {ref[k]} vs now {rm[k]}" for k in ("n_candidates", "n_val_pass", "n_test_positive", "n_test_kept"))
                + ". New trading days or revised Yahoo data since the first run can cause small differences. "
                  "The placebo comparison below uses the REAL value from this run, so it is still like-for-like.")
        if res["errors"]:
            st.warning("Tickers with problems: " + "; ".join(f"{t}: {m}" for t, m in res["errors"]))

        st.subheader(f"2. Real result vs {res['n_reps']} placebo repetitions")
        for line in pl.verdict_text(res):
            st.write("• " + line)
        summ = res["summary"]
        show = summ[["description", "real", "placebo_mean", "placebo_min", "placebo_p95", "placebo_max",
                     "reps_equal_or_exceeding_real", "empirical_p_(k+1)/(R+1)"]]
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption(f"With {res['n_reps']} repetitions the smallest possible empirical p is {1 / (res['n_reps'] + 1):.3f}. "
                   "The summary is about counts of passers, not about any single condition.")

        st.subheader("3. How the placebo counts are distributed")
        metric = st.selectbox("Metric", list(pl.METRIC_LABELS), index=list(pl.METRIC_LABELS).index("n_test_kept"),
                              format_func=lambda m: pl.METRIC_LABELS[m])
        dist = res["distribution"]
        d = dist[dist["metric"] == metric].set_index("value")["n_reps"]
        st.bar_chart(d)
        st.caption(f"Bars = how many placebo repetitions produced each value. REAL = {res['real_metrics'][metric]:.4g}.")

        st.subheader("4. Calendar boundaries of TRAIN / VALIDATION / TEST (real run)")
        st.caption("Each condition is split 60/20/20 among its own signal days, so cut dates differ between conditions.")
        st.dataframe(res["boundaries_summary"], use_container_width=True, hide_index=True)

        st.subheader("5. Checkpoint rule sensitivity (baseline hit rates, real data)")
        st.caption("'inclusive' = the engine's rule (a bar starting AT the checkpoint minute counts). "
                   "'strict' = only bars that have finished by the checkpoint.")
        st.dataframe(res["sensitivity_summary"], use_container_width=True, hide_index=True)

        st.subheader("Downloads")
        st.download_button("⬇ Everything (one ZIP: summary, every repetition, boundaries, README)", st.session_state["pl_zip"],
                           file_name="v3_placebo_results.zip", mime="application/zip", type="primary")
        d1, d2, d3 = st.columns(3)
        d1.download_button("placebo_summary.csv", res["summary"].to_csv(index=False), "placebo_summary.csv", "text/csv", key="dl_sum")
        d2.download_button("placebo_rep_metrics.csv", res["placebo_metrics"].to_csv(index=False), "placebo_rep_metrics.csv", "text/csv", key="dl_rep")
        d3.download_button("split_boundaries_real.csv", res["boundaries"].to_csv(index=False), "split_boundaries_real.csv", "text/csv", key="dl_bnd")

# ============================================================================ TAB 2
with tab2:
    st.markdown(
        "Two quick checks that do not touch the experiment:\n\n"
        "- **AUD/USD and every other driver:** when is each Yahoo daily bar actually complete in Sydney time, compared with "
        "the 9:45am scan? (Looks at the last 3 months of hourly vs daily bars.)\n"
        "- **EODHD 5-minute bars:** are they labelled by start or end time, and which label carries the opening auction?"
    )
    run_a = st.button("▶ Run timestamp audits", type="primary", disabled=not token, key="run_audit")
    if run_a:
        status = st.empty()
        conditions = rd.build_all_conditions()
        drivers = sorted({c[0] for c in conditions})
        status.info(f"Checking {len(drivers)} drivers (daily vs hourly bars)…")
        daily, hourly = pl.make_yahoo_fetchers()
        drv_tab, evidence = pl.audit_driver_timestamps(drivers, daily, hourly)
        status.info("Checking EODHD 5-minute bar labels on CBA.AU…")
        try:
            bars = pl.fetch_eodhd_sample(token)
            findings, label_tab = pl.audit_bar_labels(bars)
        except Exception as e:
            findings, label_tab = {"error": f"{type(e).__name__}: {e}"}, pd.DataFrame()
        st.session_state["au"] = (drv_tab, evidence, findings, label_tab)
        st.session_state["au_zip"] = pl.build_audit_zip(drv_tab, evidence, findings, label_tab)
        status.success("Audits finished")

    au = st.session_state.get("au")
    if au:
        drv_tab, evidence, findings, label_tab = au
        st.subheader("Driver daily-bar timing")
        st.caption("'pct_days_complete_before_0945_next_session' = share of days on which the daily bar's information was "
                   "finished before 9:45am Sydney on the ASX session the pipeline pairs it with. Anything below 100 is a look-ahead risk.")
        cols = [c for c in ["driver", "yahoo_ticker", "best_boundary", "boundary_match_rate", "last_bar_end_sydney_clock_min",
                            "last_bar_end_sydney_clock_max", "pct_days_complete_before_0945_next_session", "result"] if c in drv_tab.columns]
        st.dataframe(drv_tab[cols], use_container_width=True, hide_index=True)
        st.subheader("EODHD 5-minute bar labelling")
        st.json(findings)
        st.download_button("⬇ Audit results (ZIP)", st.session_state["au_zip"], "v3_timestamp_audits.zip", "application/zip", type="primary")

# ============================================================================ TAB 3
with tab3:
    st.markdown(pl.README.replace("\n", "  \n"))
