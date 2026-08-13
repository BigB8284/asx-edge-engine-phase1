"""
DATA VALIDATION V2 — new tickers only
=========================================
Same script, same discipline as the original data_validation.py — this
version only checks tickers NEW to the broad-expansion proposal (Gold,
Uranium, Copper, Coal, Rare Earths, Technology, Financials, REITs).
Everything already validated in the original run (S&P500, Brent, TIO=F,
LIT, URA, CCJ, BHP, GDX, WDS.AX, NST.AX, PLS.AX, BHP.AX, PDN.AX, WHC.AX
etc.) is deliberately left out here to keep this run fast and light —
it doesn't need re-checking.

Deploy as its own throwaway app/repo, same as before. Delete after use.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta

st.set_page_config(page_title="Data Validation V2", layout="centered")
st.title("Driver Ticker Validation — V2 (new tickers)")
st.caption("Checks only the tickers newly introduced by the broad-expansion proposal.")

CANDIDATES = {
    "Rare Earths ETF": ("REMX", "rare earths driver", "PRIMARY"),
    "US 10yr Treasury Yield": ("^TNX", "REITs driver — NOTE: Yahoo quotes this as yield x10 (e.g. 45.0 = 4.50%), sanity-check the scale in the output below", "PRIMARY"),
}

# ASX tickers newly introduced across the expansion themes.
ASX_SAMPLE = [
    "RMS.AX",   # Gold — proposed replacement for GOR/NEM
    "EVN.AX", "RRL.AX",  # Gold — existing basket members, not yet individually spot-checked
    "SFR.AX", "29M.AX",  # Copper
    "YAL.AX", "NHC.AX",  # Coal (WHC.AX already validated previously)
    "LYC.AX", "ILU.AX", "ARU.AX",  # Rare Earths
    "XRO.AX", "WTC.AX", "TNE.AX", "NXT.AX",  # Technology
    "CBA.AX", "WBC.AX", "NAB.AX", "ANZ.AX", "MQG.AX",  # Financials
    "GMG.AX", "SGP.AX", "SCG.AX",  # REITs
]

STALE_THRESHOLD_DAYS = 5      # last row older than this many calendar days = stale
INTERNAL_GAP_THRESHOLD_DAYS = 10  # any single gap between rows bigger than this = suspicious
HIGH_MISSING_PCT_THRESHOLD = 8.0  # % missing vs expected business days = suspicious


def check_ticker(ticker, max_retries=3):
    last_error = None
    for attempt in range(max_retries):
        try:
            hist = yf.Ticker(ticker).history(period="max")
            last_error = None
            break
        except Exception as e:
            last_error = e
            if "Too Many Requests" in str(e) or "rate limit" in str(e).lower():
                time.sleep(3 * (attempt + 1))  # back off harder each retry
                continue
            break  # a real error (not rate-limiting) - no point retrying
    else:
        hist = None

    if last_error is not None:
        return {"status": f"ERROR: {last_error}"}

    try:
        if hist.empty:
            return {"status": "EMPTY"}
        hist = hist.dropna(subset=["Close"])
        if hist.empty:
            return {"status": "EMPTY"}

        dates = pd.Series(hist.index.date)
        first_date = dates.iloc[0]
        last_date = dates.iloc[-1]
        n_rows = len(dates)

        # Expected trading days = business days (Mon-Fri) between first and
        # last date. This is a business-day approximation, not a true
        # exchange holiday calendar, so a clean ticker will still show
        # roughly 2-4% "missing" from public holidays alone — that's
        # normal, not a red flag. Flag only well above that.
        expected_days = len(pd.bdate_range(first_date, last_date))
        pct_missing = round((1 - n_rows / expected_days) * 100, 1) if expected_days else None

        # Largest internal gap between consecutive observations, in
        # calendar days. A normal weekend gap is 3 days (Fri->Mon); a
        # normal long weekend/holiday cluster might hit 4-5. Bigger than
        # that suggests a real hole in the data, not just a holiday.
        date_series = pd.to_datetime(dates)
        gaps = date_series.diff().dt.days.dropna()
        max_internal_gap = int(gaps.max()) if not gaps.empty else 0

        # Staleness: is the most recent expected trading day actually
        # present? Approximated as "most recent business day <= today".
        today = datetime.now().date()
        expected_last_bday = pd.bdate_range(end=today, periods=1)[0].date()
        days_stale = (today - last_date).days
        has_latest_session = last_date >= expected_last_bday
        is_stale = days_stale > STALE_THRESHOLD_DAYS

        flags = []
        if not has_latest_session and is_stale:
            flags.append(f"STALE ({days_stale}d behind)")
        if max_internal_gap > INTERNAL_GAP_THRESHOLD_DAYS:
            flags.append(f"GAP ({max_internal_gap}d hole)")
        if pct_missing is not None and pct_missing > HIGH_MISSING_PCT_THRESHOLD:
            flags.append(f"HIGH MISSING ({pct_missing}%)")

        return {
            "status": "OK" if not flags else "FLAGGED",
            "first_date": first_date.isoformat(),
            "last_date": last_date.isoformat(),
            "n_rows": n_rows,
            "pct_missing": pct_missing,
            "max_internal_gap_days": max_internal_gap,
            "has_latest_session": has_latest_session,
            "flags": ", ".join(flags) if flags else "—",
        }
    except Exception as e:
        return {"status": f"ERROR: {e}"}


if st.button("Run validation", type="primary", use_container_width=True):
    rows = []
    progress = st.progress(0, text="Checking drivers...")
    total = len(CANDIDATES) + len(ASX_SAMPLE)
    i = 0

    for name, (ticker, purpose, role) in CANDIDATES.items():
        result = check_ticker(ticker)
        rows.append({"name": name, "ticker": ticker, "purpose": purpose, "role": role, **result})
        i += 1
        progress.progress(i / total, text=f"Checking {name}...")

    for ticker in ASX_SAMPLE:
        result = check_ticker(ticker)
        rows.append({"name": f"ASX: {ticker}", "ticker": ticker, "purpose": "ASX outcome sample", "role": "PRIMARY", **result})
        i += 1
        progress.progress(i / total, text=f"Checking {ticker}...")

    progress.empty()
    df = pd.DataFrame(rows)

    ok_df = df[df["status"] == "OK"]
    flagged_df = df[df["status"] == "FLAGGED"]
    bad_df = df[~df["status"].isin(["OK", "FLAGGED"])]

    c1, c2, c3 = st.columns(3)
    c1.metric("Clean", len(ok_df))
    c2.metric("Flagged", len(flagged_df))
    c3.metric("Failed/empty", len(bad_df))

    if not bad_df.empty:
        st.subheader("❌ Failed or empty")
        st.dataframe(bad_df[["name", "ticker", "purpose", "role", "status"]], use_container_width=True, hide_index=True)

    if not flagged_df.empty:
        st.subheader("⚠️ Returned data, but flagged")
        st.dataframe(
            flagged_df[["name", "ticker", "role", "first_date", "last_date", "n_rows", "pct_missing", "flags"]],
            use_container_width=True, hide_index=True
        )

    st.subheader("✅ Clean")
    st.dataframe(
        ok_df[["name", "ticker", "role", "first_date", "last_date", "n_rows", "pct_missing", "max_internal_gap_days"]].sort_values("first_date"),
        use_container_width=True, hide_index=True
    )

    st.divider()
    st.download_button(
        "Download full results as CSV",
        data=df.to_csv(index=False),
        file_name="ticker_validation_results_v2.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(
        "pct_missing is measured against business days (Mon-Fri), not a true exchange "
        "holiday calendar — so 2-4% missing on a clean ticker is normal (public holidays), "
        "not a fault. Only values flagged above that threshold are called out. "
        "No fallback has been auto-substituted anywhere above — PRIMARY and FALLBACK "
        "candidates are both shown so you can pick."
    )
else:
    st.info("Tap 'Run validation' to check all candidate tickers.")
