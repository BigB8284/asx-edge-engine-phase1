"""
ASX UNIVERSE AUDIT - core logic (no Streamlit)
==============================================
Finds the largest practical universe of tradeable ASX ordinary shares.

Pipeline (cheap first, expensive last):
  1. Symbol list       exchange-symbol-list/AU            1 call
  2. Daily screen      eod-bulk-last-day/AU per session   100 calls x ~62 sessions (cached forever)
  3. 5-minute check    intraday 5m, ONLY for daily passers at the low ($500k) bar
                       ~4 requests x 5 calls per stock for history from Oct 2020 (cached forever)
  4. Results           one row per symbol, eligibility at $1M and $500k, every reason shown

Reused from the existing repo (not reimplemented):
  intraday_data._fetch_one_window   - the 5m request with retry/backoff
  intraday_data.build_clean_day_groups - the SAME completeness / continuous-session /
                                         implausible-move pipeline the V3 engine uses, so
                                         "usable sessions" here = sessions the engine can use
  eodhd_logic.to_sydney_and_classify - DST-correct Sydney time

One deliberate difference from intraday_data.fetch_full_intraday_history:
windows here OVERLAP by one boundary instant and are de-duplicated. The
original steps window_end + 1 day, which skips most of the ASX session on
each window-end date (ASX trades ~23:00-05:00 UTC). The original is left
untouched because the frozen research depends on it.

Every run is resumable: progress is saved per stock, so pressing Run again
continues where the last run stopped (time budget, API quota, failures).
"""
import hashlib
import io
import json
import math
import re
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import intraday_data
from eodhd_logic import to_sydney_and_classify

SYDNEY = ZoneInfo("Australia/Sydney")
EXCHANGE = "AU"
BASE = "https://eodhd.com/api"

# EODHD documented cost per request (api-limits page): EOD/symbols 1,
# bulk whole exchange 100, intraday 5, /api/user 0.
CALL_COST = {"symbols": 1, "bulk": 100, "intraday": 5, "user": 0}
MAX_DAYS_PER_INTRADAY_REQUEST = 600

DEFAULTS = {
    "min_price": 1.0,
    "turnover_main": 1_000_000,
    "turnover_low": 500_000,
    "lookback_sessions": 60,
    "min_traded_share": 0.90,
    "min_usable_5m_sessions": 400,
    "min_new_listing_sessions": 20,      # below this a new listing is WATCH, re-checked next audit
    "suspended_no_trade_sessions": 5,    # no trade in the last N sessions -> suspended/inactive
    "min_morning_coverage": 0.80,        # share of 10:00-11:00 5m slots that traded
    "quality_recent_sessions": 60,
    "intraday_start": "2020-10-01",      # EODHD non-US 5m history begins Oct 2020
    "bulk_ready_hour_sydney": 19,        # EODHD posts ASX EOD ~2-3h after the close
}

NON_ORDINARY_NAME = re.compile(
    r"\b(RIGHTS?|OPTIONS?|WARRANTS?|DEFERRED|CONVERTIBLE|PREFERENCE|PREF|HYBRID|"
    r"CAPITAL NOTES?|NOTES?|BONDS?|ETF|ETP|INSTALMENTS?|DELISTED)\b", re.I)
ORDINARY_CODE = re.compile(r"^[A-Z0-9]{3}$")

# ASX moved to a single open on 23 June 2025 (SR15), so every stock opens
# ~10:00 and the full first hour is a fair test. The quality window only
# looks at the last 60 sessions, all after that change.
MORNING_SLOTS_START = "10:00"
MORNING_SLOTS_END = "11:00"
N_MORNING_SLOTS = 12            # 5m bars starting 10:00, 10:05, ..., 10:55
OPEN_BAR_END = "10:05"          # the 10:00 bar holds the opening auction

KEY_SYMBOLS = "meta/symbols_AU.parquet"
KEY_BULK = "meta/daily_bulk_AU.parquet"
KEY_BULK_SKIPPED = "meta/bulk_non_trading_dates.json"
KEY_QUALITY = "meta/intraday_quality.parquet"
KEY_RUNLOG = "meta/run_log.json"


def bars_key(code):
    return f"bars5m/{code}.parquet"


class QuotaExhausted(RuntimeError):
    """EODHD daily limit (or persistent rate limit) hit: stop cleanly, resume later."""


class FetchError(RuntimeError):
    pass


def params_with_defaults(p=None):
    out = dict(DEFAULTS)
    out.update(p or {})
    return out


def _df_to_parquet_bytes(df):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="zstd")
    return buf.getvalue()


def _parquet_bytes_to_df(b):
    return pd.read_parquet(io.BytesIO(b))


# ==========================================================================
# EODHD client (daily endpoints + call accounting)
# ==========================================================================
class EODHDClient:
    def __init__(self, token, session=None, sleep=time.sleep):
        self.token = token
        self.s = session or requests.Session()
        self.sleep = sleep
        self.calls = {k: 0 for k in CALL_COST}
        self.requests_made = {k: 0 for k in CALL_COST}

    def total_calls(self):
        return sum(self.calls.values())

    def _count(self, kind, n_requests=1):
        self.requests_made[kind] += n_requests
        self.calls[kind] += CALL_COST[kind] * n_requests

    def _get(self, path, params, kind, retries=4):
        params = {**params, "api_token": self.token, "fmt": "json"}
        last = "unknown"
        for attempt in range(retries):
            try:
                r = self.s.get(f"{BASE}/{path}", params=params, timeout=60)
            except requests.RequestException as e:
                last = f"network error: {e}"
                self.sleep(2 * (attempt + 1))
                continue
            self._count(kind)
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    last = f"non-JSON response: {r.text[:200]}"
                    self.sleep(2)
                    continue
            if r.status_code == 402:
                raise QuotaExhausted(f"EODHD says the API limit is used up (HTTP 402): {r.text[:200]}")
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", "0") or 0) or 5 * (attempt + 1)
                last = f"HTTP 429: {r.text[:200]}"
                self.sleep(min(wait, 60))
                continue
            if r.status_code in (401, 403):
                raise FetchError(f"EODHD rejected the token (HTTP {r.status_code}). Check the secret.")
            if r.status_code == 404:
                return None
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            self.sleep(2 * (attempt + 1))
        if "429" in last:
            raise QuotaExhausted(f"EODHD kept rate-limiting ({last}). Likely the daily limit - resume later.")
        raise FetchError(f"{path} failed after {retries} tries: {last}")

    def user(self):
        try:
            return self._get("user", {}, "user") or {}
        except Exception as e:  # free, informational - never fatal
            return {"error": str(e)}

    def symbols(self):
        data = self._get(f"exchange-symbol-list/{EXCHANGE}", {}, "symbols")
        if not data:
            raise FetchError("EODHD returned an empty ASX symbol list.")
        return pd.DataFrame(data)

    def bulk_day(self, date_str):
        data = self._get(f"eod-bulk-last-day/{EXCHANGE}", {"date": date_str}, "bulk")
        return pd.DataFrame(data or [])

    def _window(self, ticker, w0, w1):
        """One 5m request via the existing intraday_data fetcher (retry/backoff included)."""
        data, reason = intraday_data._fetch_one_window(ticker, self.token, w0, w1)
        self._count("intraday")
        if data is None and reason and ("402" in reason or "429" in reason):
            raise QuotaExhausted(f"{ticker}: {reason}")
        return data, reason

    def intraday_history(self, code, start_dt, end_dt):
        """Full 5m history, paginated with overlapping windows. Returns
        (bars_df, failed_windows). Raises QuotaExhausted on 402/429 so the run
        stops cleanly instead of burning through failures."""
        ticker = f"{code}.{EXCHANGE}"
        frames, failed = [], []
        w0 = start_dt
        span = timedelta(days=MAX_DAYS_PER_INTRADAY_REQUEST - 1)
        while w0 < end_dt:
            w1 = min(w0 + span, end_dt)
            data, reason = self._window(ticker, w0, w1)
            if data is None:
                # one retry as two half-size windows (big responses can time out)
                mid = w0 + (w1 - w0) / 2
                halves = [self._window(ticker, w0, mid), self._window(ticker, mid, w1)]
                if all(h[0] is not None for h in halves):
                    data = halves[0][0] + halves[1][0]
                else:
                    failed.append((str(w0.date()), str(w1.date()), next(h[1] for h in halves if h[0] is None)))
            if data:
                frames.append(pd.DataFrame(data))
            if w1 >= end_dt:
                break
            w0 = w1  # overlap one instant; duplicates removed below
        return _clean_raw_bars(frames), failed


def _clean_raw_bars(frames):
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    if not frames:
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True)
    if "timestamp" not in df.columns and "datetime" in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"], utc=True).astype("int64") // 10**9
    df = df[[c for c in cols if c in df.columns]].copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].astype("int64")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


# ==========================================================================
# Stage 1-2: symbols + daily bulk sessions
# ==========================================================================
def latest_candidate_date(now_syd, ready_hour):
    d = now_syd.date() if now_syd.hour >= ready_hour else now_syd.date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def collect_daily_sessions(client, store, p, now_syd, log=print, max_requests=110):
    """Returns (bulk_df for the last `lookback_sessions` sessions, sessions list).
    Past sessions are cached forever; holidays are remembered so they're
    never re-requested. A date only counts as a session if the bulk rows
    returned actually carry that date (EODHD may answer a holiday with the
    previous day's data)."""
    n_needed = int(p["lookback_sessions"])
    cached = store.get(KEY_BULK)
    bulk = _parquet_bytes_to_df(cached) if cached else pd.DataFrame()
    skipped = set(json.loads(store.get(KEY_BULK_SKIPPED) or b"[]"))
    have = set(bulk["date"].unique()) if not bulk.empty else set()

    d = latest_candidate_date(now_syd, int(p["bulk_ready_hour_sydney"]))
    first_candidate = d
    sessions, requests_used, new_frames, dirty = [], 0, [], False
    while len(sessions) < n_needed:
        ds = d.isoformat()
        if ds in have:
            sessions.append(ds)
        elif ds not in skipped:
            if requests_used >= max_requests:
                log(f"Stopped after {max_requests} bulk requests with {len(sessions)} sessions found.")
                break
            df = client.bulk_day(ds)
            requests_used += 1
            if not df.empty and "date" in df.columns:
                df = df[df["date"].astype(str) == ds]
            if len(df) >= 300:
                keep = [c for c in ("code", "date", "open", "high", "low", "close", "adjusted_close", "volume")
                        if c in df.columns]
                df = df[keep].copy()
                df["date"] = ds
                new_frames.append(df)
                have.add(ds)
                sessions.append(ds)
                dirty = True
            elif d == first_candidate:
                log(f"{ds}: EODHD has no complete bulk file yet (normal before ~7pm Sydney) - using earlier sessions.")
            else:
                skipped.add(ds)
                dirty = True
            if dirty and len(new_frames) and len(new_frames) % 10 == 0:
                bulk = _save_bulk(store, bulk, new_frames, skipped)
                new_frames = []
                dirty = False
        d -= timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
    if dirty or new_frames:
        bulk = _save_bulk(store, bulk, new_frames, skipped)
    sessions = sorted(sessions)
    window = bulk[bulk["date"].isin(sessions)].copy() if not bulk.empty else bulk
    return window, sessions, requests_used


def _save_bulk(store, bulk, new_frames, skipped, keep_sessions=300):
    if new_frames:
        bulk = pd.concat([bulk] + new_frames, ignore_index=True) if not bulk.empty else pd.concat(new_frames, ignore_index=True)
        bulk = bulk.drop_duplicates(["code", "date"], keep="last")
    if bulk.empty:
        store.put(KEY_BULK_SKIPPED, json.dumps(sorted(skipped)).encode())
        return bulk
    dates = sorted(bulk["date"].unique())
    if len(dates) > keep_sessions:
        bulk = bulk[bulk["date"].isin(dates[-keep_sessions:])]
    store.put(KEY_BULK, _df_to_parquet_bytes(bulk.reset_index(drop=True)))
    store.put(KEY_BULK_SKIPPED, json.dumps(sorted(skipped)).encode())
    return bulk


# ==========================================================================
# Stage 2: daily screen (pure function - unit tested)
# ==========================================================================
def screen_daily(symbols, bulk, sessions, p):
    """One row per symbol in the ASX list. Daily gates only; the 5m gates
    are applied later in build_results."""
    p = params_with_defaults(p)
    sym = symbols.copy()
    for c in ("Code", "Name", "Type", "Currency", "Isin", "Exchange", "Country"):
        if c not in sym.columns:
            sym[c] = None
    sym = sym.drop_duplicates("Code")
    n_sess = len(sessions)

    if bulk is not None and not bulk.empty:
        b = bulk[bulk["date"].isin(sessions)].copy()
        b["close"] = pd.to_numeric(b["close"], errors="coerce")
        b["volume"] = pd.to_numeric(b["volume"], errors="coerce").fillna(0)
        vol = b.pivot_table(index="code", columns="date", values="volume", aggfunc="last").reindex(columns=sessions)
        close = b.pivot_table(index="code", columns="date", values="close", aggfunc="last").reindex(columns=sessions)
    else:
        vol = close = pd.DataFrame(columns=sessions)

    rows = []
    last_n_suspend = int(p["suspended_no_trade_sessions"])
    for _, s in sym.iterrows():
        code = str(s["Code"]).strip().upper()
        r = {"code": code, "ticker_ax": f"{code}.AX", "ticker_eodhd": f"{code}.{EXCHANGE}",
             "name": s["Name"], "type": s["Type"], "currency": s["Currency"], "isin": s["Isin"]}
        hard = []  # reasons that apply at any turnover threshold
        if str(s["Type"]).strip().lower() != "common stock":
            hard.append(f"not_ordinary_share({s['Type']})")
        if s["Currency"] not in (None, "", "AUD") and not (isinstance(s["Currency"], float) and math.isnan(s["Currency"])):
            hard.append(f"non_aud_quote({s['Currency']})")
        if not ORDINARY_CODE.match(code):
            hard.append("code_not_ordinary_3_char")
        if isinstance(s["Name"], str) and NON_ORDINARY_NAME.search(s["Name"]):
            hard.append("name_suggests_non_ordinary")

        if code in vol.index:
            v = vol.loc[code].to_numpy(dtype=float)
            c = close.loc[code].to_numpy(dtype=float)
            present = ~np.isnan(c)
            first_idx = int(np.argmax(present)) if present.any() else n_sess
            if 0 < first_idx < 5:
                first_idx = 0  # missing only from the first few sessions = didn't trade, not a new listing
            elig = n_sess - first_idx            # sessions since first seen in the window
            v_w = np.nan_to_num(v[first_idx:], nan=0.0)
            c_w = c[first_idx:]
            traded = v_w > 0
            c_ffill = pd.Series(c_w).ffill().to_numpy()
            turnover = np.where(traded, np.nan_to_num(c_ffill) * v_w, 0.0)
            last_trade_pos = np.where(traded)[0]
            sessions_since_trade = (elig - 1 - last_trade_pos[-1]) if len(last_trade_pos) else elig
            last_close = c_ffill[-1] if len(c_ffill) else np.nan
            r.update({
                "daily_sessions_observed": int(elig),
                "first_session_in_window": sessions[first_idx] if elig else None,
                "is_new_listing": bool(first_idx > 0),
                "last_close": float(last_close) if not np.isnan(last_close) else None,
                "traded_sessions": int(traded.sum()),
                "traded_share": round(float(traded.mean()), 4) if elig else 0.0,
                "median_turnover_aud": round(float(np.median(turnover)), 0) if elig else 0.0,
                "mean_turnover_aud": round(float(np.mean(turnover)), 0) if elig else 0.0,
                "latest_turnover_aud": round(float(turnover[-1]), 0) if elig else 0.0,
                "last_trade_date": sessions[first_idx + last_trade_pos[-1]] if len(last_trade_pos) else None,
                "sessions_since_last_trade": int(sessions_since_trade),
                "in_latest_session": bool(present[-1]) if n_sess else False,
            })
            if not hard:
                if len(last_trade_pos) == 0 or sessions_since_trade >= last_n_suspend or not present[-1]:
                    hard.append(f"suspended_or_inactive(no trade in last {last_n_suspend} sessions)")
                if r["last_close"] is None or r["last_close"] < p["min_price"]:
                    hard.append(f"price_below_{p['min_price']:g}")
                if elig >= p["min_new_listing_sessions"] and r["traded_share"] < p["min_traded_share"]:
                    hard.append(f"traded_share_below_{int(p['min_traded_share'] * 100)}pct")
        else:
            r.update({"daily_sessions_observed": 0, "first_session_in_window": None, "is_new_listing": False,
                      "last_close": None, "traded_sessions": 0, "traded_share": 0.0,
                      "median_turnover_aud": 0.0, "mean_turnover_aud": 0.0, "latest_turnover_aud": 0.0,
                      "last_trade_date": None, "sessions_since_last_trade": None, "in_latest_session": False})
            if not hard:
                hard.append("no_daily_data_in_window")

        watch_new = (not hard) and r["daily_sessions_observed"] < p["min_new_listing_sessions"]
        r["daily_hard_reasons"] = "; ".join(hard)
        r["new_listing_watch"] = bool(watch_new)
        for label, thr in (("1m", p["turnover_main"]), ("500k", p["turnover_low"])):
            ok = (not hard) and (not watch_new) and r["median_turnover_aud"] >= thr
            r[f"daily_pass_{label}"] = bool(ok)
        rows.append(r)
    return pd.DataFrame(rows)


# ==========================================================================
# Stage 3: 5-minute assessment (pure function - unit tested)
# ==========================================================================
def assess_intraday(bars, p):
    """Metrics from raw 5m bars. 'usable_5m_sessions' uses the V3 engine's
    own cleaning (build_clean_day_groups), so it is exactly what the model
    could train on."""
    p = params_with_defaults(p)
    out = {"bars_total": int(len(bars)), "first_5m_date": None, "last_5m_date": None,
           "raw_5m_sessions": 0, "usable_5m_sessions": 0, "incomplete_days": 0, "flagged_move_days": 0,
           "morning_coverage": None, "open_bar_traded_share": None, "median_morning_turnover_aud": None,
           "median_continuous_bars_per_day": None, "quality_note": ""}
    if bars is None or bars.empty:
        out["quality_note"] = "no 5m bars returned"
        return out
    records = bars.to_dict("records")
    clean_days, excluded, flagged, _ = intraday_data.build_clean_day_groups(records)
    df, err = to_sydney_and_classify(records)
    if err:
        out["quality_note"] = err
        return out
    cont = df[df["in_continuous_session"]].copy()
    day_list = sorted(cont["sydney_date"].unique())
    out.update({
        "first_5m_date": str(day_list[0]) if day_list else None,
        "last_5m_date": str(day_list[-1]) if day_list else None,
        "raw_5m_sessions": len(day_list),
        "usable_5m_sessions": len(clean_days),
        "incomplete_days": int(len(excluded)) if excluded is not None else 0,
        "flagged_move_days": int(len(flagged)) if flagged is not None else 0,
    })
    if not day_list:
        out["quality_note"] = "no continuous-session bars"
        return out
    recent = set(day_list[-int(p["quality_recent_sessions"]):])
    rc = cont[cont["sydney_date"].isin(recent)].copy()
    rc["hhmm"] = rc["sydney_dt"].dt.strftime("%H:%M")
    rc["vol"] = pd.to_numeric(rc["volume"], errors="coerce").fillna(0)
    rc["px"] = pd.to_numeric(rc["close"], errors="coerce")
    morning = rc[(rc["hhmm"] >= MORNING_SLOTS_START) & (rc["hhmm"] < MORNING_SLOTS_END) & (rc["vol"] > 0)]
    per_day_slots = morning.groupby("sydney_date")["hhmm"].nunique().reindex(sorted(recent), fill_value=0)
    out["morning_coverage"] = round(float((per_day_slots / N_MORNING_SLOTS).clip(upper=1).mean()), 4)
    opening = rc[(rc["hhmm"] >= "10:00") & (rc["hhmm"] < OPEN_BAR_END) & (rc["vol"] > 0)]
    out["open_bar_traded_share"] = round(opening["sydney_date"].nunique() / len(recent), 4)
    m_all = rc[(rc["hhmm"] >= "10:00") & (rc["hhmm"] < MORNING_SLOTS_END)]
    mt = (m_all["px"] * m_all["vol"]).groupby(m_all["sydney_date"]).sum().reindex(sorted(recent), fill_value=0)
    out["median_morning_turnover_aud"] = round(float(mt.median()), 0)
    out["median_continuous_bars_per_day"] = float(cont.groupby("sydney_date").size().median())
    return out


def _load_quality(store):
    b = store.get(KEY_QUALITY)
    return _parquet_bytes_to_df(b) if b else pd.DataFrame(columns=["code"])


def _save_quality(store, q):
    store.put(KEY_QUALITY, _df_to_parquet_bytes(q.reset_index(drop=True)))


def process_intraday_ticker(code, client, store, p, now_utc):
    """Fetch (or reuse cached) 5m history for one stock, store raw bars,
    return the quality row. Cached stocks cost 0 calls (+1 request top-up
    if the cache is more than a week old)."""
    start = datetime.fromisoformat(p["intraday_start"]).replace(tzinfo=timezone.utc)
    key = bars_key(code)
    cached = store.get(key)
    calls_before = client.calls["intraday"]
    failed = []
    source = "fetched"
    if cached:
        bars = _parquet_bytes_to_df(cached)
        source = "cache"
        last_ts = int(bars["timestamp"].max()) if not bars.empty else None
        if last_ts is not None and now_utc - datetime.fromtimestamp(last_ts, timezone.utc) > timedelta(days=7):
            top, failed = client.intraday_history(code, datetime.fromtimestamp(last_ts, timezone.utc), now_utc)
            if not top.empty:
                bars = _clean_raw_bars([bars, top])
                store.put(key, _df_to_parquet_bytes(bars))
                source = "cache+topup"
    else:
        bars, failed = client.intraday_history(code, start, now_utc)
        if failed:
            return {"code": code, "fetch_status": "FAILED", "failed_windows": len(failed),
                    "fetch_error": str(failed[0][2])[:300], "assessed_at_utc": now_utc.isoformat(),
                    "intraday_calls": client.calls["intraday"] - calls_before}
        store.put(key, _df_to_parquet_bytes(bars))
    q = assess_intraday(bars, p)
    q.update({"code": code, "fetch_status": "OK", "source": source, "failed_windows": len(failed),
              "fetch_error": str(failed[0][2])[:300] if failed else "",
              "parquet_bytes": store.size(key) if not bars.empty else 0,
              "assessed_at_utc": now_utc.isoformat(),
              "intraday_calls": client.calls["intraday"] - calls_before})
    return q


# ==========================================================================
# Stage 4: results (pure function - unit tested)
# ==========================================================================
def build_results(daily, quality, p, version):
    p = params_with_defaults(p)
    q = quality.drop_duplicates("code", keep="last").set_index("code") if not quality.empty else pd.DataFrame()
    res = daily.copy()
    qcols = ["fetch_status", "first_5m_date", "last_5m_date", "raw_5m_sessions", "usable_5m_sessions",
             "incomplete_days", "flagged_move_days", "morning_coverage", "open_bar_traded_share",
             "median_morning_turnover_aud", "median_continuous_bars_per_day", "bars_total",
             "parquet_bytes", "quality_note", "fetch_error", "assessed_at_utc"]
    for c in qcols:
        res[c] = res["code"].map(q[c]) if (not q.empty and c in q.columns) else None

    for label, thr in (("1m", p["turnover_main"]), ("500k", p["turnover_low"])):
        statuses, reasons, live, train = [], [], [], []
        for _, r in res.iterrows():
            why = [x for x in str(r["daily_hard_reasons"] or "").split("; ") if x]
            st = None
            if why:
                st = "EXCLUDED"
            elif r["new_listing_watch"]:
                st, why = "WATCH_NEW_LISTING", [f"only {r['daily_sessions_observed']} sessions since listing - re-check next audit"]
            elif r["median_turnover_aud"] < thr:
                st, why = "EXCLUDED", [f"median_turnover_below_{int(thr):,}"]
            elif r["fetch_status"] is None or (isinstance(r["fetch_status"], float) and math.isnan(r["fetch_status"])):
                st, why = "PENDING_5M_CHECK", ["5m data not checked yet - press Run again"]
            elif r["fetch_status"] == "FAILED":
                st, why = "PENDING_5M_CHECK", [f"5m fetch failed, will retry: {r['fetch_error']}"]
            else:
                cov = r["morning_coverage"]
                usable = r["usable_5m_sessions"]
                if cov is None or (isinstance(cov, float) and math.isnan(cov)):
                    st, why = "EXCLUDED", [f"no usable 5m data ({r['quality_note'] or 'empty'})"]
                elif cov < p["min_morning_coverage"]:
                    st, why = "EXCLUDED", [f"thin_morning_trading(coverage {cov:.0%} < {p['min_morning_coverage']:.0%})"]
                elif usable is None or (isinstance(usable, float) and math.isnan(usable)) or usable == 0:
                    st, why = "EXCLUDED", ["no clean 5m sessions (every day failed the completeness check)"]
                elif usable < p["min_usable_5m_sessions"]:
                    st, why = "LIVE_ONLY_BUILDING_HISTORY", [
                        f"{int(r['usable_5m_sessions'])} usable 5m sessions < {p['min_usable_5m_sessions']} - "
                        "covered live, joins training once history builds"]
                else:
                    st, why = "ELIGIBLE", []
            statuses.append(st)
            reasons.append("; ".join(why))
            live.append(st in ("ELIGIBLE", "LIVE_ONLY_BUILDING_HISTORY"))
            train.append(st == "ELIGIBLE")
        res[f"status_{label}"] = statuses
        res[f"reasons_{label}"] = reasons
        res[f"live_coverage_{label}"] = live
        res[f"training_eligible_{label}"] = train

    res.insert(0, "audit_version", version)
    order = {"ELIGIBLE": 0, "LIVE_ONLY_BUILDING_HISTORY": 1, "PENDING_5M_CHECK": 2, "WATCH_NEW_LISTING": 3, "EXCLUDED": 4}
    res["_o"] = res["status_1m"].map(order).fillna(9) * 10 + res["status_500k"].map(order).fillna(9)
    res = res.sort_values(["_o", "median_turnover_aud"], ascending=[True, False]).drop(columns="_o")
    lead = ["audit_version", "code", "ticker_ax", "name", "status_1m", "status_500k", "reasons_1m", "reasons_500k",
            "live_coverage_1m", "training_eligible_1m", "live_coverage_500k", "training_eligible_500k",
            "last_close", "median_turnover_aud", "traded_share", "usable_5m_sessions", "morning_coverage"]
    return res[lead + [c for c in res.columns if c not in lead]].reset_index(drop=True)


def summarise(res):
    out = {}
    for label in ("1m", "500k"):
        vc = res[f"status_{label}"].value_counts()
        out[label] = {k: int(vc.get(k, 0)) for k in
                      ("ELIGIBLE", "LIVE_ONLY_BUILDING_HISTORY", "PENDING_5M_CHECK", "WATCH_NEW_LISTING", "EXCLUDED")}
    return out


def exclusion_breakdown(res, label):
    ex = res[res[f"status_{label}"] == "EXCLUDED"][f"reasons_{label}"].fillna("")
    first = ex.str.split("; ").str[0].str.replace(r"\(.*\)", "", regex=True)
    return first.value_counts().rename_axis("first_reason").reset_index(name="stocks")


# ==========================================================================
# Estimates: download, API, storage, daily running cost
# ==========================================================================
def estimate_costs(res, quality, p, user_info=None, now_utc=None):
    p = params_with_defaults(p)
    now_utc = now_utc or datetime.now(timezone.utc)
    start = datetime.fromisoformat(p["intraday_start"]).replace(tzinfo=timezone.utc)
    hist_days = (now_utc - start).days
    req_per_stock = max(1, math.ceil(hist_days / (MAX_DAYS_PER_INTRADAY_REQUEST - 1)))
    sessions_est = int(np.busday_count(start.date(), now_utc.date()) * 0.97)

    q_ok = quality[quality.get("fetch_status", pd.Series(dtype=object)) == "OK"] if not quality.empty else quality
    measured = (not q_ok.empty) and q_ok["bars_total"].fillna(0).sum() > 0
    full_hist = q_ok[q_ok["raw_5m_sessions"].fillna(0) >= 400] if measured else q_ok
    bars_per_stock = (float(full_hist["bars_total"].median()) if measured and not full_hist.empty
                      else sessions_est * 74.0)  # new listings excluded: they'd understate a full history
    bytes_per_bar = (float(q_ok["parquet_bytes"].fillna(0).sum()) / max(1.0, float(q_ok["bars_total"].fillna(0).sum()))
                     if measured else 12.0)
    json_bytes_per_bar = 115.0  # EODHD 5m JSON row ~110-120 bytes (estimate)

    need = res[res["daily_pass_500k"]]
    pending = need[res.loc[need.index, "status_500k"] == "PENDING_5M_CHECK"]
    n_need, n_pending = len(need), len(pending)
    live_1m = int(res["live_coverage_1m"].sum())
    live_500k = int(res["live_coverage_500k"].sum())

    def mb(x):
        return round(x / 1e6, 1)

    est = {
        "history_from": p["intraday_start"],
        "history_requests_per_stock": req_per_stock,
        "history_calls_per_stock": req_per_stock * CALL_COST["intraday"],
        "stocks_needing_5m": n_need,
        "stocks_still_pending_5m": n_pending,
        "calls_to_finish_5m": n_pending * req_per_stock * CALL_COST["intraday"],
        "calls_full_5m_from_scratch": n_need * req_per_stock * CALL_COST["intraday"],
        "daily_screen_calls_first_run": 62 * CALL_COST["bulk"] + CALL_COST["symbols"],
        "bars_per_stock": int(bars_per_stock),
        "bars_per_stock_basis": "measured" if measured else "estimated",
        "download_mb_remaining": mb(n_pending * bars_per_stock * json_bytes_per_bar),
        "download_mb_full": mb(n_need * bars_per_stock * json_bytes_per_bar),
        "storage_mb_full": mb(n_need * bars_per_stock * bytes_per_bar),
        "storage_bytes_per_bar": round(bytes_per_bar, 1),
        "live_universe_1m": live_1m,
        "live_universe_500k": live_500k,
    }
    for label, n in (("1m", live_1m), ("500k", live_500k)):
        est[f"daily_update_calls_{label}"] = CALL_COST["bulk"] + n * CALL_COST["intraday"] + CALL_COST["symbols"]
        est[f"daily_storage_growth_mb_{label}"] = round(n * 74 * bytes_per_bar / 1e6, 2)
        est[f"yearly_storage_growth_mb_{label}"] = mb(n * 74 * bytes_per_bar * 252)
    if user_info and "dailyRateLimit" in user_info:
        lim = int(user_info.get("dailyRateLimit") or 0) + int(user_info.get("extraLimit") or 0)
        est["plan_daily_limit"] = lim
        est["used_today"] = int(user_info.get("apiRequests") or 0)
        est["remaining_today"] = max(0, lim - est["used_today"])
    return est


# ==========================================================================
# Orchestrator (resumable)
# ==========================================================================
def run_audit(client, store, p=None, time_budget_s=20 * 60, progress=None, log=print, now_utc=None,
              reserve_calls=2_000):
    """Runs/resumes the full audit. Returns a dict with results, summary,
    estimates and run stats. Never raises for API trouble - it stops
    cleanly and says why."""
    p = params_with_defaults(p)
    t0 = time.time()
    now_utc = now_utc or datetime.now(timezone.utc)
    now_syd = now_utc.astimezone(SYDNEY)
    progress = progress or (lambda frac, msg: None)
    stop_reason = None

    user_before = client.user()
    if "dailyRateLimit" in user_before:
        log(f"EODHD plan: {user_before.get('apiRequests', '?')} of {user_before.get('dailyRateLimit')} calls used today.")

    progress(0.02, "Fetching ASX symbol list...")
    try:
        symbols = client.symbols()
        store.put(KEY_SYMBOLS, _df_to_parquet_bytes(symbols))
    except QuotaExhausted as e:
        b = store.get(KEY_SYMBOLS)
        if not b:
            return {"ok": False, "stop_reason": str(e)}
        symbols, stop_reason = _parquet_bytes_to_df(b), str(e)
        log("Quota hit - using the cached symbol list.")
    n_common = int((symbols["Type"] == "Common Stock").sum()) if "Type" in symbols.columns else 0
    log(f"{len(symbols)} ASX symbols; {n_common} listed as Common Stock.")

    progress(0.05, "Collecting daily sessions (cached after first run)...")
    try:
        bulk, sessions, bulk_req = collect_daily_sessions(client, store, p, now_syd, log=log)
    except QuotaExhausted as e:
        return {"ok": False, "stop_reason": f"API quota hit during the daily screen: {e}"}
    log(f"Daily screen uses {len(sessions)} sessions ({sessions[0] if sessions else '-'} to "
        f"{sessions[-1] if sessions else '-'}); {bulk_req} new bulk requests this run.")
    if len(sessions) < p["lookback_sessions"]:
        log(f"WARNING: only {len(sessions)} sessions available (wanted {p['lookback_sessions']}).")

    daily = screen_daily(symbols, bulk, sessions, p)
    need = daily[daily["daily_pass_500k"]].sort_values("median_turnover_aud", ascending=False)
    quality = _load_quality(store)
    done = set(quality.loc[quality.get("fetch_status", pd.Series(dtype=object)) == "OK", "code"]) if not quality.empty else set()
    todo = [c for c in need["code"] if c not in done]
    log(f"{len(need)} stocks pass the daily screen at $500k; {len(done & set(need['code']))} already 5m-checked; "
        f"{len(todo)} to check now (largest turnover first).")

    rows, since_save = [], 0
    for i, code in enumerate(todo):
        if time.time() - t0 > time_budget_s:
            stop_reason = f"Time budget reached after {i} of {len(todo)} stocks - press Run again to continue."
            break
        lim = user_before.get("dailyRateLimit")
        if lim and int(user_before.get("apiRequests") or 0) + client.total_calls() > int(lim) - reserve_calls:
            stop_reason = "Stopped to keep a reserve of the daily EODHD limit - resume tomorrow (or raise the reserve)."
            break
        progress(0.1 + 0.85 * i / max(1, len(todo)), f"5m check {i + 1}/{len(todo)}: {code}")
        try:
            row = process_intraday_ticker(code, client, store, p, now_utc)
        except QuotaExhausted as e:
            stop_reason = f"EODHD limit hit at {code}: {e}. Progress saved - resume later."
            break
        except Exception as e:  # one bad stock never kills the run
            row = {"code": code, "fetch_status": "FAILED", "fetch_error": f"{type(e).__name__}: {e}"[:300],
                   "assessed_at_utc": now_utc.isoformat()}
        rows.append(row)
        since_save += 1
        if since_save >= 10:
            quality = _merge_quality(quality, rows)
            _save_quality(store, quality)
            rows, since_save = [], 0
    if rows:
        quality = _merge_quality(quality, rows)
        _save_quality(store, quality)

    as_of = sessions[-1] if sessions else now_syd.date().isoformat()
    phash = hashlib.sha1(json.dumps({k: p[k] for k in sorted(p)}, default=str).encode()).hexdigest()[:6]
    version = f"v{as_of.replace('-', '')}_{now_syd.strftime('%H%M')}_{phash}"
    res = build_results(daily, quality, p, version)
    summary = summarise(res)
    user_after = client.user()
    est = estimate_costs(res, quality, p, user_after if "dailyRateLimit" in user_after else user_before, now_utc)
    complete = summary["500k"]["PENDING_5M_CHECK"] == 0 and stop_reason is None
    csv_name = f"asx_universe_audit_{version}{'' if complete else '_PARTIAL'}.csv"
    csv_bytes = res.to_csv(index=False).encode()
    meta = {"version": version, "as_of_session": as_of, "sessions_used": len(sessions),
            "complete": complete, "stop_reason": stop_reason, "params": p, "summary": summary,
            "estimates": est, "calls_this_run": dict(client.calls), "total_calls_this_run": client.total_calls(),
            "seconds": round(time.time() - t0, 1), "run_at_utc": now_utc.isoformat(), "csv_name": csv_name,
            "storage": store.describe()}
    try:
        store.put(f"results/{csv_name}", csv_bytes)
        store.put("results/latest.csv", csv_bytes)
        store.put("results/latest_meta.json", json.dumps(meta, default=str, indent=2).encode())
    except Exception as e:
        log(f"WARNING: could not save results to storage ({e}) - download the CSV now.")
    progress(1.0, "Done" if complete else "Stopped - partial results")
    return {"ok": True, "results": res, "meta": meta, "csv_name": csv_name, "csv_bytes": csv_bytes,
            "complete": complete, "stop_reason": stop_reason, "summary": summary, "estimates": est,
            "quality": quality}


def _merge_quality(quality, rows):
    new = pd.DataFrame(rows)
    if quality is None or quality.empty:
        return new
    return pd.concat([quality, new], ignore_index=True).drop_duplicates("code", keep="last")


def load_latest(store):
    b = store.get("results/latest.csv")
    m = store.get("results/latest_meta.json")
    if not b or not m:
        return None
    return {"results": pd.read_csv(io.BytesIO(b)), "meta": json.loads(m), "csv_bytes": b}
