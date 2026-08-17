"""
V3 PERSISTENCE — save cleaned raw 5-minute bars so we never re-fetch
================================================================
Per explicit instruction: persist the cleaned raw bars this time, do
NOT fetch -> calculate -> discard. GitHub is NOT chosen as the
permanent store yet (that decision is still open) — for this stage,
data is saved locally (survives across runs within a deployment's
uptime, though not guaranteed across a redeploy/long sleep) AND made
available as a direct download, so Brent can archive it externally
himself in the meantime.

Also persists exclusion logs (incomplete days, implausible moves,
missing-price bars) per ticker, so a full audit trail survives
alongside the bars themselves — not just the clean data.

Storage format: Parquet (columnar, ~7x smaller than CSV for this kind
of numeric OHLCV data — see the size estimate given earlier: ~78MB for
the full ~100-ticker universe at Parquet compression, trivially small).
"""

import os
import json
import pandas as pd

DATA_DIR = "v3_raw_data"
BARS_SUBDIR = "bars"
EXCLUSIONS_SUBDIR = "exclusions"


def _ensure_dirs():
    os.makedirs(os.path.join(DATA_DIR, BARS_SUBDIR), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, EXCLUSIONS_SUBDIR), exist_ok=True)


def _bars_path(ticker):
    return os.path.join(DATA_DIR, BARS_SUBDIR, f"{ticker.replace('.', '_')}.parquet")


def _exclusions_path(ticker):
    return os.path.join(DATA_DIR, EXCLUSIONS_SUBDIR, f"{ticker.replace('.', '_')}.json")


def has_persisted_data(ticker):
    return os.path.exists(_bars_path(ticker))


def save_ticker_data(ticker, clean_days, excluded_days, flagged_moves, outside_info, failed_windows):
    """clean_days: dict of {date_str: DataFrame of that day's bars}, as
    returned by intraday_data.build_clean_day_groups. Flattens to one
    Parquet file per ticker (a 'trading_date' column preserves the
    per-day grouping so it can be reconstructed on load) plus a JSON
    sidecar with everything needed for a full audit trail: which days
    were excluded and why, flagged implausible moves, DST/session
    classification notes, and any fetch-window failures.
    """
    _ensure_dirs()

    if clean_days:
        frames = []
        for date_str, day_bars in clean_days.items():
            day_bars = day_bars.copy()
            day_bars["trading_date"] = date_str
            frames.append(day_bars)
        combined = pd.concat(frames, ignore_index=True)
        combined.to_parquet(_bars_path(ticker), index=False)

    exclusions_record = {
        "ticker": ticker,
        "n_clean_days": len(clean_days) if clean_days else 0,
        "excluded_days": excluded_days,       # e.g. incomplete-day exclusions, with reasons
        "flagged_moves": flagged_moves,       # implausible-move flags
        "outside_info": outside_info,         # DST/session classification notes
        "failed_fetch_windows": failed_windows,
    }
    with open(_exclusions_path(ticker), "w") as f:
        json.dump(exclusions_record, f, indent=2, default=str)


def load_ticker_data(ticker):
    """Returns (clean_days, exclusions_record) reconstructed from disk,
    or (None, None) if nothing is persisted for this ticker yet."""
    if not has_persisted_data(ticker):
        return None, None

    combined = pd.read_parquet(_bars_path(ticker))
    clean_days = {
        date_str: day_df.drop(columns=["trading_date"]).reset_index(drop=True)
        for date_str, day_df in combined.groupby("trading_date")
    }

    exclusions_record = None
    if os.path.exists(_exclusions_path(ticker)):
        with open(_exclusions_path(ticker)) as f:
            exclusions_record = json.load(f)

    return clean_days, exclusions_record


def get_or_fetch(ticker, fetch_fn):
    """fetch_fn: zero-arg callable returning (clean_days, excluded_days,
    flagged_moves, outside_info, failed_windows) — i.e. the same
    signature as load_intraday_for_ticker in the pilot app. Checks disk
    first; only calls fetch_fn (which hits EODHD) on a genuine miss.
    Always saves after a fresh fetch, so the SECOND time this ticker is
    ever needed — in this pilot, in the full-universe run later, in a
    re-analysis with a different threshold — it's already there.
    """
    cached_days, cached_exclusions = load_ticker_data(ticker)
    if cached_days is not None:
        return cached_days, cached_exclusions, True  # True = served from cache

    clean_days, excluded_days, flagged_moves, outside_info, failed_windows = fetch_fn()
    save_ticker_data(ticker, clean_days, excluded_days, flagged_moves, outside_info, failed_windows)
    exclusions_record = {
        "ticker": ticker, "n_clean_days": len(clean_days) if clean_days else 0,
        "excluded_days": excluded_days, "flagged_moves": flagged_moves,
        "outside_info": outside_info, "failed_fetch_windows": failed_windows,
    }
    return clean_days, exclusions_record, False  # False = freshly fetched


def export_all_as_zip(zip_path="v3_raw_data_export.zip"):
    """Bundles the entire persisted dataset (all tickers' bars +
    exclusion logs) into a single zip for manual download/archival —
    the explicit export path requested in place of auto-committing to
    GitHub at this stage."""
    import shutil
    if not os.path.exists(DATA_DIR):
        return None
    base = zip_path[:-4] if zip_path.endswith(".zip") else zip_path
    shutil.make_archive(base, "zip", DATA_DIR)
    return f"{base}.zip"


def storage_summary():
    """Returns a small dict describing what's currently persisted —
    ticker count, total size, oldest/newest fetch — for display in the
    app so it's always visible what's cached vs what would need a
    fresh fetch."""
    _ensure_dirs()
    bars_dir = os.path.join(DATA_DIR, BARS_SUBDIR)
    tickers = [f[:-8].replace("_", ".") for f in os.listdir(bars_dir) if f.endswith(".parquet")]
    total_bytes = sum(
        os.path.getsize(os.path.join(bars_dir, f))
        for f in os.listdir(bars_dir) if f.endswith(".parquet")
    )
    return {
        "n_tickers_cached": len(tickers),
        "tickers_cached": sorted(tickers),
        "total_size_mb": round(total_bytes / 1_000_000, 1),
    }
