"""
ASX UNIVERSE AUDIT - Streamlit app (one button)
===============================================
Main file path on Streamlit Cloud: universe_audit_app.py

Secrets (Streamlit -> app -> Settings -> Secrets):
    EODHD_API_TOKEN = "..."              # same token as the other apps
    DATA_REPO       = "BigB8284/asx-data-store"   # PRIVATE repo for the cache
    DATA_REPO_TOKEN = "github_pat_..."   # fine-grained token, Contents read/write on that repo only
Without DATA_REPO/DATA_REPO_TOKEN it still runs, but on temporary disk.
"""
import json
import os

import streamlit as st

import universe_audit_core as core
from data_store import StoreError, make_store

st.set_page_config(page_title="ASX Universe Audit", layout="centered")
st.title("ASX Universe Audit")
st.caption("Finds every ASX ordinary share that's liquid and clean enough for the predictive scanner. "
           "Daily screen first, 5-minute data only for stocks that pass. Every stock and every exclusion "
           "reason is in the CSV.")


def _secret(*names):
    try:
        for n in names:
            if n in st.secrets and st.secrets[n]:
                return str(st.secrets[n])
    except Exception:
        pass
    return os.environ.get(names[0])


def get_token():
    # Same pattern as placebo_app / reverse_discovery_app: any secret key containing "EODHD".
    try:
        for k in st.secrets:
            if "EODHD" in str(k).upper() and isinstance(st.secrets[k], str):
                return st.secrets[k]
        for k in ("api_token", "API_TOKEN"):
            if k in st.secrets:
                return st.secrets[k]
    except Exception:
        pass
    return os.environ.get("EODHD_API_TOKEN")


FAKE = bool(os.environ.get("UNIVERSE_AUDIT_FAKE"))  # test hook only
token = "fake" if FAKE else get_token()
if not token:
    st.error("No EODHD token found in Streamlit secrets (add EODHD_API_TOKEN).")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_store(repo, repo_token):
    return make_store(repo, repo_token)


try:
    store, store_warning = get_store(_secret("DATA_REPO"), _secret("DATA_REPO_TOKEN", "GITHUB_DATA_TOKEN"))
except StoreError as e:
    st.error(f"Permanent storage is configured but not working: {e}")
    st.info("Fix the DATA_REPO / DATA_REPO_TOKEN secrets (or remove both to run on temporary disk).")
    st.stop()

store.refresh()  # pick up anything a GitHub Actions run wrote since the last page load
if store_warning:
    st.warning(store_warning)
else:
    st.success(f"Cache: {store.describe()} - survives redeploys.", icon="💾")

with st.expander("Settings (defaults = your agreed rules)"):
    c1, c2 = st.columns(2)
    min_price = c1.number_input("Min price (A$)", value=core.DEFAULTS["min_price"], step=0.5)
    turnover_main = c1.number_input("Main turnover bar (A$/day)", value=core.DEFAULTS["turnover_main"], step=100_000)
    turnover_low = c1.number_input("Comparison bar (A$/day)", value=core.DEFAULTS["turnover_low"], step=100_000)
    min_traded = c1.number_input("Min share of sessions traded", value=core.DEFAULTS["min_traded_share"], step=0.01)
    min_hist = c2.number_input("Min usable 5m sessions (training)", value=core.DEFAULTS["min_usable_5m_sessions"], step=50)
    min_cov = c2.number_input("Min 10:00-11:00 bar coverage", value=core.DEFAULTS["min_morning_coverage"], step=0.05)
    budget_min = c2.number_input("Stop this run after (minutes)", value=20, min_value=2, max_value=120, step=5,
                                 help="Long first runs are split into chunks. Press Run again to continue - nothing is re-downloaded.")
    reserve = c2.number_input("Keep this many EODHD calls spare", value=2_000, step=1_000)
params = core.params_with_defaults({
    "min_price": float(min_price), "turnover_main": float(turnover_main), "turnover_low": float(turnover_low),
    "min_traded_share": float(min_traded), "min_usable_5m_sessions": int(min_hist),
    "min_morning_coverage": float(min_cov)})

run = st.button("▶ Run / resume audit", type="primary", width="stretch")

if run:
    if FAKE:
        from tests.fakes import make_fake_client
        client = make_fake_client()
    else:
        client = core.EODHDClient(token)
    status = st.empty()
    bar = st.progress(0.0)
    log_box = st.expander("Run log", expanded=False)
    lines = []

    def progress(frac, msg):
        bar.progress(min(1.0, max(0.0, frac)))
        status.info(msg)

    def log(msg):
        lines.append(msg)
        log_box.write(msg)

    try:
        out = core.run_audit(client, store, params, time_budget_s=int(budget_min) * 60,
                             progress=progress, log=log, reserve_calls=int(reserve))
    except core.FetchError as e:
        st.error(f"EODHD problem: {e}")
        st.stop()
    except StoreError as e:
        st.error(f"Storage problem: {e}")
        st.stop()
    if not out.get("ok"):
        st.error(out.get("stop_reason", "Audit stopped."))
        st.stop()
    st.session_state["audit"] = {"results": out["results"], "meta": out["meta"], "csv_bytes": out["csv_bytes"]}
    st.session_state["audit_log"] = lines
    status.empty()
    bar.empty()

audit = st.session_state.get("audit")
if audit is None:
    try:
        audit = core.load_latest(store)
    except Exception as e:
        st.warning(f"Couldn't load the last saved audit: {e}")
        audit = None
    if audit:
        st.session_state["audit"] = audit

if not audit:
    st.info("No audit yet. Press Run - the first run takes several chunks (5m history for a few hundred stocks).")
    st.stop()

res, meta = audit["results"], audit["meta"]
summ = meta["summary"]

if meta.get("complete"):
    st.success(f"Audit complete - version {meta['version']} (data to {meta['as_of_session']}).")
else:
    st.warning(f"PARTIAL audit - {meta.get('stop_reason') or 'more stocks to check'} "
               f"({summ['500k']['PENDING_5M_CHECK']} stocks still to 5m-check). Press Run again.")

st.subheader("Eligible universe")
c1, c2 = st.columns(2)
for col, label, title in ((c1, "1m", "$1M/day bar"), (c2, "500k", "$500k/day bar")):
    s = summ[label]
    col.metric(f"{title}: eligible", s["ELIGIBLE"],
               help="Passes every rule incl. >=400 usable 5m sessions - goes into initial model training.")
    col.metric("+ live-only (building history)", s["LIVE_ONLY_BUILDING_HISTORY"],
               help="Liquid and clean, but newer: scanned live, joins training once it has 400 sessions.")
    col.caption(f"Pending 5m check {s['PENDING_5M_CHECK']} · new-listing watch {s['WATCH_NEW_LISTING']} · "
                f"excluded {s['EXCLUDED']}")

csv_name = meta.get("csv_name", "asx_universe_audit.csv")
st.download_button("⬇ Download full audit CSV (every stock)", audit["csv_bytes"], file_name=csv_name,
                   mime="text/csv", width="stretch")

tab1, tab2, tab3, tab4 = st.tabs(["Eligible", "Exclusions", "Costs & storage", "Run details"])

with tab1:
    show = st.radio("Show", ["$1M eligible + live-only", "Extra stocks gained at $500k"], horizontal=True)
    if show.startswith("$1M"):
        view = res[res["live_coverage_1m"].astype(bool)]
    else:
        view = res[res["live_coverage_500k"].astype(bool) & ~res["live_coverage_1m"].astype(bool)]
    cols = ["code", "name", "status_1m", "last_close", "median_turnover_aud", "traded_share",
            "usable_5m_sessions", "morning_coverage", "median_morning_turnover_aud"]
    st.dataframe(view[[c for c in cols if c in view.columns]], hide_index=True, width="stretch",
                 column_config={"median_turnover_aud": st.column_config.NumberColumn("median $/day", format="%.0f"),
                                "median_morning_turnover_aud": st.column_config.NumberColumn("median $ 10-11am", format="%.0f"),
                                "morning_coverage": st.column_config.NumberColumn("10-11am coverage", format="%.2f")})

with tab2:
    for label, title in (("1m", "$1M bar"), ("500k", "$500k bar")):
        st.markdown(f"**Why stocks were excluded ({title}, first reason per stock)**")
        st.dataframe(core.exclusion_breakdown(res, label), hide_index=True, width="stretch")
    st.caption("Newer listings are never permanently excluded: under 20 sessions they're on WATCH, "
               "under 400 usable 5m sessions they're LIVE-ONLY. Both are re-checked every audit.")

with tab3:
    e = meta["estimates"]
    st.markdown("**Historical 5-minute download**")
    st.write(f"- History from {e['history_from']}: {e['history_requests_per_stock']} requests = "
             f"{e['history_calls_per_stock']} API calls per stock")
    st.write(f"- {e['stocks_needing_5m']} stocks need 5m data · {e['stocks_still_pending_5m']} still to fetch "
             f"(~{e['calls_to_finish_5m']:,} calls, ~{e['download_mb_remaining']} MB download)")
    st.write(f"- Full set from scratch: ~{e['calls_full_5m_from_scratch']:,} calls, ~{e['download_mb_full']} MB download")
    st.write(f"- ~{e['bars_per_stock']:,} bars per stock ({e['bars_per_stock_basis']})")
    st.markdown("**Storage (compressed Parquet)**")
    st.write(f"- Full 5m history: ~{e['storage_mb_full']} MB ({e['storage_bytes_per_bar']} bytes/bar)")
    st.write(f"- Growth: ~{e['daily_storage_growth_mb_1m']} MB/day ($1M universe), "
             f"~{e['yearly_storage_growth_mb_1m']} MB/year · $500k universe ~{e['yearly_storage_growth_mb_500k']} MB/year")
    st.markdown("**Expected daily update cost**")
    st.write(f"- $1M universe ({e['live_universe_1m']} stocks): ~{e['daily_update_calls_1m']:,} calls/day")
    st.write(f"- $500k universe ({e['live_universe_500k']} stocks): ~{e['daily_update_calls_500k']:,} calls/day")
    st.caption("Daily update = 1 bulk EOD file (100 calls) + one 5m request per stock (5 calls) + symbol list (1).")
    if "plan_daily_limit" in e:
        st.markdown("**Your EODHD plan today**")
        st.write(f"- {e['used_today']:,} of {e['plan_daily_limit']:,} calls used · {e['remaining_today']:,} left")
    calls = meta.get("calls_this_run", {})
    st.write(f"**This run used ~{meta.get('total_calls_this_run', 0):,} calls** "
             f"(symbols {calls.get('symbols', 0)}, bulk {calls.get('bulk', 0)}, 5m {calls.get('intraday', 0)})")

with tab4:
    st.write(f"Version `{meta['version']}` · run at {meta['run_at_utc'][:16]} UTC · {meta['seconds']} s · "
             f"{meta['sessions_used']} daily sessions · storage: {meta.get('storage')}")
    st.json(meta["params"], expanded=False)
    st.download_button("⬇ Run details (JSON)", json.dumps(meta, indent=2, default=str),
                       file_name=csv_name.replace(".csv", "_meta.json"), mime="application/json")
    if st.session_state.get("audit_log"):
        st.text("\n".join(st.session_state["audit_log"]))
