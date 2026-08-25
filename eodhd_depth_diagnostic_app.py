"""
EODHD INTRADAY DEPTH DIAGNOSTIC — Streamlit app
==================================================
Diagnostic only. Does not touch the scanner, the reverse-discovery
methodology, any thresholds, or the pilot universe. Answers exactly one
question: how much REAL, USABLE 5-minute intraday history does EODHD
actually have for the 16 V3 pilot tickers, using the SAME fetch method
and SAME quality pipeline the live V3 intraday system uses —
intraday_data.fetch_full_intraday_history() and
intraday_data.build_clean_day_groups(), both imported unmodified. This
tool does not reimplement or approximate that logic.

For each ticker, reports BOTH:
  - RAW stats: earliest/latest bar EODHD actually returns, raw unique
    session count, raw total bar count — i.e. what's in the API
    response before any quality filtering.
  - CLEAN stats: unique sessions and bar count AFTER the exact same
    completeness/continuous-session/implausible-move filtering the
    live system applies — i.e. what's actually usable.
Both are shown because they answer different questions: RAW shows what
EODHD has; CLEAN shows what reverse discovery could actually use.

Also reports the common overlapping CLEAN date range across all
tickers that fetched successfully, and the number of common clean
sessions in that overlap — the real answer to "how much matched
history is available."

Failed fetch windows and per-ticker exclusions are shown explicitly,
never silently dropped, per the project's standing rule.
"""
import time
from datetime import datetime, date

import streamlit as st
import pandas as pd

from intraday_data import fetch_full_intraday_history, build_clean_day_groups
from eodhd_logic import to_sydney_and_classify

# The 16 approved pilot tickers, from v3_pilot_config.PILOT_TICKERS.
# Editable below in case the roster changes — this default is just what
# was confirmed as of 2026-08-25.
DEFAULT_TICKERS = [
    "CIA.AX", "ILU.AX", "ARU.AX", "NXT.AX", "GMG.AX",
    "BPT.AX", "SFR.AX", "WDS.AX", "BOE.AX",
    "CBA.AX", "QBE.AX", "S32.AX",
    "JHX.AX", "NEM.AX", "BHP.AX", "RIO.AX",
]

st.set_page_config(page_title="EODHD Intraday Depth Diagnostic", layout="wide")
st.title("EODHD Intraday Depth Diagnostic")
st.caption(
    "Diagnostic only — no scanner, methodology, threshold, or universe changes. "
    "Uses the real intraday_data.fetch_full_intraday_history() and "
    "build_clean_day_groups() exactly as they exist in the V3 system."
)

def _default_api_token():
    # st.secrets raises if no secrets.toml exists at all (not just a
    # plain KeyError on a missing key), so this has to be a try/except,
    # not a hasattr check — the no-secrets-configured case is exactly
    # the one this fallback needs to survive.
    try:
        return st.secrets.get("EODHD_API_TOKEN", "")
    except Exception:
        return ""


api_token = st.text_input(
    "EODHD API token",
    value=_default_api_token(),
    type="password",
    help="Not stored anywhere by this app. Prefer setting EODHD_API_TOKEN in Streamlit secrets "
         "instead of pasting it here each time.",
)
tickers_text = st.text_area(
    "Pilot tickers (one per line) — confirm/edit as needed",
    value="\n".join(DEFAULT_TICKERS),
    height=220,
)
col1, col2 = st.columns(2)
with col1:
    start_date_input = st.date_input("Search from (widen this if 'earliest bar' below equals this date)", value=date(2015, 1, 1))
with col2:
    end_date_input = st.date_input("Search to", value=date.today())

st.warning(
    "If any ticker's 'raw earliest bar' below equals your chosen start date exactly, "
    "that's a sign there may be more history before this point — widen the start date and rerun.",
    icon="⚠️",
)

if st.button("Run EODHD depth diagnostic", type="primary"):
    if not api_token:
        st.error("Enter an EODHD API token first.")
        st.stop()

    tickers = [t.strip() for t in tickers_text.splitlines() if t.strip()]
    if not tickers:
        st.error("No tickers entered.")
        st.stop()

    start_dt = datetime.combine(start_date_input, datetime.min.time())
    end_dt = datetime.combine(end_date_input, datetime.min.time())

    progress = st.progress(0.0)
    status = st.empty()

    rows = []
    clean_date_sets = {}
    all_failed_windows = []
    all_excluded_days = {}
    all_flagged_moves = {}

    for i, ticker in enumerate(tickers):
        status.write(f"Fetching {ticker}...  [{i + 1}/{len(tickers)}]")
        progress.progress((i + 1) / len(tickers))

        raw_bars, failed_windows = fetch_full_intraday_history(ticker, api_token, start_dt, end_dt)
        for w_start, w_end, reason in failed_windows:
            all_failed_windows.append({"ticker": ticker, "window_start": w_start, "window_end": w_end, "reason": reason})

        if not raw_bars:
            rows.append({
                "ticker": ticker,
                "raw_earliest_bar": None, "raw_latest_bar": None,
                "raw_unique_sessions": 0, "raw_total_bars": 0,
                "clean_unique_sessions": 0, "clean_total_bars": 0,
                "excluded_days": 0, "flagged_implausible_days": 0,
                "n_failed_windows": len(failed_windows),
                "note": "No bars returned at all — check token/ticker/date range" if not failed_windows
                        else "No bars returned, and some windows failed — see failed-windows table",
            })
            clean_date_sets[ticker] = set()
            continue

        raw_df, err = to_sydney_and_classify(raw_bars)
        if err:
            rows.append({
                "ticker": ticker,
                "raw_earliest_bar": None, "raw_latest_bar": None,
                "raw_unique_sessions": 0, "raw_total_bars": len(raw_bars),
                "clean_unique_sessions": 0, "clean_total_bars": 0,
                "excluded_days": 0, "flagged_implausible_days": 0,
                "n_failed_windows": len(failed_windows),
                "note": f"Response format error: {err}",
            })
            clean_date_sets[ticker] = set()
            continue

        clean_days, excluded_days, flagged_moves, outside_info = build_clean_day_groups(raw_bars)
        all_excluded_days[ticker] = excluded_days
        all_flagged_moves[ticker] = flagged_moves
        clean_date_sets[ticker] = set(clean_days.keys())

        rows.append({
            "ticker": ticker,
            "raw_earliest_bar": str(raw_df["sydney_dt"].min()),
            "raw_latest_bar": str(raw_df["sydney_dt"].max()),
            "raw_unique_sessions": raw_df["sydney_date"].nunique(),
            "raw_total_bars": len(raw_df),
            "clean_unique_sessions": len(clean_days),
            "clean_total_bars": sum(len(d) for d in clean_days.values()),
            "excluded_days": len(excluded_days),
            "flagged_implausible_days": len(flagged_moves),
            "n_failed_windows": len(failed_windows),
            "note": "",
        })

        time.sleep(0.3)  # gentle pacing between tickers, not part of the real fetch function itself

    status.write("Done.")
    df = pd.DataFrame(rows)
    st.subheader("Per-ticker results")
    st.dataframe(df, width="stretch")
    st.download_button(
        "Download per-ticker results CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="eodhd_depth_diagnostic_results.csv",
        mime="text/csv",
    )

    st.subheader("Common overlap across successfully-fetched tickers (CLEAN sessions only)")
    non_empty_sets = {t: s for t, s in clean_date_sets.items() if s}
    if len(non_empty_sets) < len(tickers):
        missing = [t for t in tickers if t not in non_empty_sets]
        st.warning(f"{len(missing)} ticker(s) had zero clean sessions and are excluded from the overlap calc: {missing}")
    if non_empty_sets:
        common = set.intersection(*non_empty_sets.values())
        if common:
            common_sorted = sorted(common)
            st.write(f"**{len(common)} common clean trading sessions** across {len(non_empty_sets)} tickers, "
                     f"from **{common_sorted[0]}** to **{common_sorted[-1]}**.")
        else:
            st.write("No common clean sessions across all fetched tickers.")
    else:
        st.write("No ticker returned any clean sessions — nothing to overlap.")

    if all_failed_windows:
        st.subheader("Failed fetch windows (shown, not silently dropped)")
        st.dataframe(pd.DataFrame(all_failed_windows), width="stretch")

    with st.expander("Per-ticker excluded (incomplete) days and flagged implausible-move days"):
        for t in tickers:
            exc = all_excluded_days.get(t)
            flg = all_flagged_moves.get(t)
            if exc is not None and not exc.empty:
                st.write(f"**{t} — excluded (incomplete) days:** {len(exc)}")
                st.dataframe(exc, width="stretch")
            if flg is not None and not flg.empty:
                st.write(f"**{t} — flagged implausible-move days:** {len(flg)}")
                st.dataframe(flg, width="stretch")
