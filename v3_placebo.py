"""
V3 PLACEBO CALIBRATION — core logic (no Streamlit in this file)
==================================================================
Purpose: find out whether the reverse-discovery results (152 TRAIN
candidates, 67 validation passes, 34 Test-kept) are stronger than what
RANDOMISED overnight data produces when pushed through the identical
discovery -> validation -> test process.

This module REUSES the original engine (v3_reverse_discovery.py) and
changes nothing in it. The original app, code, results and CSVs are
untouched. Everything here is additive.

How the placebo works
---------------------
For each placebo repetition, every ticker's ALIGNED driver table (rows =
that ticker's own ASX sessions, columns = the drivers, exactly as the real
run builds it) is rolled by k sessions as ONE block:

  * all drivers move together  -> driver-to-driver correlations are
    preserved exactly;
  * each driver's distribution, volatility clustering and its own
    missing-value pattern are preserved exactly (the rows are the same
    rows, just attached to different ASX sessions);
  * the roll is circular INSIDE the ticker's own data window, so no row is
    created, dropped, or pulled from outside the window (no missing data
    is introduced and nothing from beyond the end of the data is used);
  * k is never small: at least MIN_SHIFT_SESSIONS (126, about six months)
    away from the true alignment in either direction, so short-lag
    relationships and local regimes cannot survive;
  * every ticker in a repetition uses the SAME relative shift, so stocks
    that react to the same overnight days keep sharing the same placebo
    signal days (preserves cross-stock clustering of chance results);
  * seeds are fixed: repetition r always draws the same shift for a given
    base seed, so 20 reps are exactly the first 20 of 100.

Each repetition then re-runs the COMPLETE process: all 156 signals x
2 directions x 4 targets x 3 primary checkpoints, one TRAIN-selected
candidate per combination, validation with the locked rule, test for
validation-passers. No condition is pre-selected.

Repetition 0 is the REAL (unshifted) data pushed through the same code,
so the app proves it reproduces the original run before comparing.
"""
import copy
import hashlib
import json
import math
import os
import pickle
import time

import numpy as np
import pandas as pd

import v3_reverse_discovery as rd

# ---------------------------------------------------------------------------
# FIXED PARAMETERS (not tunable from the app)
# ---------------------------------------------------------------------------
MIN_SHIFT_SESSIONS = 126          # ~6 months of ASX sessions, minimum distance from true alignment
DEFAULT_SEED = 20261116
CONVENTIONS = ("original", "strict")   # original = engine's '<= checkpoint minute'; strict = '< checkpoint minute'
CACHE_HOURS = 20

# Real-run reference (from rd_1_candidates_validation_test.csv, run 2026-11-16)
REAL_REFERENCE = {
    "n_candidates": 152,
    "n_val_pass": 67,
    "n_test_positive": 48,
    "n_test_kept": 34,
    "n_val_pass_conditions": 35,
    "n_test_kept_conditions": 18,
    "n_test_kept_nonthin": 10,
    "n_test_kept_nonthin_conditions": 3,
    "max_test_uplift_kept": 21.5,
    "mean_test_uplift_kept": 13.76,
}

METRIC_LABELS = {
    "n_candidates": "TRAIN candidates (of 384 combinations)",
    "n_val_pass": "Passed validation",
    "n_test_positive": "Passed validation and Test uplift positive",
    "n_test_kept": "Passed validation, Test positive and kept >=50% of TRAIN uplift  [headline]",
    "n_val_pass_conditions": "Distinct conditions among validation passers",
    "n_test_kept_conditions": "Distinct conditions among Test-kept rows",
    "n_test_kept_nonthin": "Test-kept rows with >=30 days in BOTH validation and Test",
    "n_test_kept_nonthin_conditions": "Distinct conditions among those non-THIN rows",
    "max_test_uplift_kept": "Largest Test uplift among Test-kept rows (pp)",
    "mean_test_uplift_kept": "Mean Test uplift among Test-kept rows (pp)",
}
HEADLINE = ["n_candidates", "n_val_pass", "n_test_kept"]
FAM = ["ticker", "direction", "driver", "driver_threshold_pct", "driver_side"]


# ---------------------------------------------------------------------------
# CHECKPOINT CONVENTION (sensitivity switch; default = exactly the original)
# ---------------------------------------------------------------------------
def apply_convention(td, mode):
    """'original': unchanged (engine counts a bar whose START minute <= checkpoint minute).
    'strict': counts only bars that START before the checkpoint minute, i.e. data that has
    finished by the checkpoint (bars are labelled by start time in EODHD). Built from the
    engine's own first-crossing times, so nothing is re-simulated."""
    if mode == "original":
        return td
    if mode != "strict":
        raise ValueError(mode)
    new = copy.copy(td)
    new.hit = {}
    for d in rd.DIRECTIONS:
        t = td.t_min[d]
        h = np.zeros_like(td.hit[d])
        for ci, cp in enumerate(rd.CHECKPOINTS):
            h[:, ci, :] = (~np.isnan(t)) & (t < rd.CP_MINUTES[cp])
        new.hit[d] = h
    return new


def inclusive_hits_from_tmin(td, direction):
    """Re-derive the engine's reached_by_checkpoint from first-crossing times (used to prove equivalence)."""
    t = td.t_min[direction]
    h = np.zeros_like(td.hit[direction])
    for ci, cp in enumerate(rd.CHECKPOINTS):
        h[:, ci, :] = (~np.isnan(t)) & (t <= rd.CP_MINUTES[cp])
    return h


# ---------------------------------------------------------------------------
# SHIFTS
# ---------------------------------------------------------------------------
def draw_u(seed, rep):
    """Fixed, reproducible draw in [0,1) for repetition `rep` (independent of how many reps run)."""
    return float(np.random.default_rng(np.random.SeedSequence([int(seed), int(rep)])).random())


def shift_sessions(n, u, min_shift=MIN_SHIFT_SESSIONS):
    """Shift k in [min_shift, n - min_shift]; same relative position u for every ticker."""
    hi = n - min_shift
    if hi < min_shift:
        raise ValueError(f"only {n} sessions: cannot keep a {min_shift}-session minimum shift both ways")
    return int(min_shift + math.floor(u * (hi - min_shift + 1)))


def roll_aligned(aligned, k):
    """Circular roll of the whole aligned driver block: row i receives original row (i+k) mod n.
    Index (the ticker's real ASX session dates) is unchanged."""
    vals = np.roll(aligned.to_numpy(), -k, axis=0)
    return pd.DataFrame(vals, index=aligned.index, columns=aligned.columns)


# ---------------------------------------------------------------------------
# FULL DISCOVERY -> VALIDATION -> TEST, lean (no matrix/continuation/gap rows)
# ---------------------------------------------------------------------------
def discover_validate_test(td, aligned, conditions, splits=None):
    """Same logic, same locked rules and same engine helpers as
    v3_reverse_discovery.process_ticker, without building the ungated matrix or
    descriptive tables (not needed for counting, and 20x slower). Returns a list of
    one row per (direction, target, checkpoint) combination (384 across 16 tickers)."""
    if splits is None:
        splits = rd.build_signal_splits(aligned, conditions, td.n)
    out = []
    for direction in rd.DIRECTIONS:
        H = td.hit[direction]
        stats = {}
        for key, sp in splits.items():
            if len(sp.train) == 0:
                stats[key] = rd.train_stats(np.zeros((0, 3, 4), bool), np.zeros((0, 3, 4), bool))
            else:
                stats[key] = rd.train_view_stats(H, sp)
        chosen = rd.select_candidate(stats)
        for cp in rd.PRIMARY_CHECKPOINTS:
            for t in rd.TARGETS:
                ch = chosen[(cp, t)]
                key = ch["candidate"]
                row = {"ticker": td.ticker, "direction": direction, "target_pct": t, "checkpoint": cp,
                       "n_signals_eligible_on_train": ch["n_eligible"]}
                if key is None:
                    row["status"] = "NOT_A_CANDIDATE"
                    out.append(row)
                    continue
                sp = splits[key]
                tr = rd._split_block(td, direction, sp, "train", cp, t)
                va = rd._split_block(td, direction, sp, "validation", cp, t)
                passed, reason, retention = rd.validation_verdict(tr["uplift_pp"], va["uplift_pp"], va["n_signal"])
                row.update({
                    "driver": key[0], "driver_threshold_pct": key[1], "driver_side": key[2],
                    "n_train": tr["n_signal"], "n_validation": va["n_signal"], "n_test": len(sp.test),
                    "train_signal_prob": round(tr["signal_prob"], 1), "train_baseline_prob": round(tr["baseline_prob"], 1),
                    "train_uplift_pp": round(tr["uplift_pp"], 1),
                    "val_signal_prob": _r1(va["signal_prob"]), "val_baseline_prob": _r1(va["baseline_prob"]),
                    "val_uplift_pp": _r1(va["uplift_pp"]),
                    "val_retention_of_train": None if retention is None else round(retention, 2),
                    "validation_passed": passed, "reason": reason,
                    "status": "VALIDATION_PASSED" if passed else "FAILED_VALIDATION",
                })
                if passed:
                    te = rd._split_block(td, direction, sp, "test", cp, t)
                    tret = (te["uplift_pp"] / tr["uplift_pp"]) if not math.isnan(te["uplift_pp"]) else None
                    row.update({
                        "n_test": te["n_signal"], "test_signal_prob": _r1(te["signal_prob"]),
                        "test_baseline_prob": _r1(te["baseline_prob"]), "test_uplift_pp": _r1(te["uplift_pp"]),
                        "test_retention_of_train": None if tret is None else round(tret, 2),
                        "test_same_direction_positive": bool(not math.isnan(te["uplift_pp"]) and te["uplift_pp"] > 0),
                        "test_retains_50pct_of_train": bool(tret is not None and tret >= rd.VALIDATION_RETENTION),
                    })
                out.append(row)
    return out


def _r1(x):
    return None if (x is None or (isinstance(x, float) and math.isnan(x))) else round(float(x), 1)


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------
def rep_metrics(combos):
    """Counts comparable with the real run's 152 / 67 / 34 (same definitions)."""
    c = combos
    cand = c[c["status"] != "NOT_A_CANDIDATE"]
    p = c[c["status"] == "VALIDATION_PASSED"]
    if len(p):
        pos = p["test_same_direction_positive"].fillna(False).astype(bool)
        keep = pos & p["test_retains_50pct_of_train"].fillna(False).astype(bool)
        kept = p[keep]
        nonthin = kept[(kept["n_validation"] >= 30) & (kept["n_test"] >= 30)]
    else:
        pos = pd.Series([], dtype=bool)
        kept = p
        nonthin = p
    return {
        "n_candidates": int(len(cand)),
        "n_val_pass": int(len(p)),
        "n_test_positive": int(pos.sum()) if len(p) else 0,
        "n_test_kept": int(len(kept)),
        "n_val_pass_conditions": int(p.groupby(FAM).ngroups) if len(p) else 0,
        "n_test_kept_conditions": int(kept.groupby(FAM).ngroups) if len(kept) else 0,
        "n_test_kept_nonthin": int(len(nonthin)),
        "n_test_kept_nonthin_conditions": int(nonthin.groupby(FAM).ngroups) if len(nonthin) else 0,
        "max_test_uplift_kept": float(kept["test_uplift_pp"].max()) if len(kept) else 0.0,
        "mean_test_uplift_kept": float(kept["test_uplift_pp"].mean()) if len(kept) else 0.0,
    }


def compare_to_placebo(real, placebo_df):
    """One row per metric: real value vs the placebo distribution."""
    R = len(placebo_df)
    rows = []
    for m, label in METRIC_LABELS.items():
        v = placebo_df[m].to_numpy(dtype=float)
        rv = float(real[m])
        ge = int((v >= rv - 1e-9).sum())
        rows.append({
            "metric": m, "description": label, "real": round(rv, 2),
            "placebo_reps": R, "placebo_mean": round(float(v.mean()), 2), "placebo_median": round(float(np.median(v)), 2),
            "placebo_min": round(float(v.min()), 2), "placebo_p05": round(float(np.percentile(v, 5)), 2),
            "placebo_p95": round(float(np.percentile(v, 95)), 2), "placebo_max": round(float(v.max()), 2),
            "reps_equal_or_exceeding_real": ge, "share_equal_or_exceeding_real": round(ge / R, 3),
            "empirical_p_(k+1)/(R+1)": round((ge + 1) / (R + 1), 4),
            "real_over_placebo_mean": (round(rv / float(v.mean()), 2) if v.mean() > 0 else None),
        })
    return pd.DataFrame(rows)


def distribution_table(placebo_df):
    rows = []
    for m in METRIC_LABELS:
        vc = placebo_df[m].round(0 if m.startswith("n_") else 1).value_counts().sort_index()
        for val, n in vc.items():
            rows.append({"metric": m, "value": val, "n_reps": int(n)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SPLIT BOUNDARIES (calendar dates) -- the real run, documented
# ---------------------------------------------------------------------------
def boundaries_rows(td, splits):
    d = td.dates

    def dt(i):
        return str(d[i].date()) if (i is not None and 0 <= i < td.n) else None

    rows = []
    for key, sp in splits.items():
        rows.append({
            "ticker": td.ticker, "driver": key[0], "driver_threshold_pct": key[1], "driver_side": key[2],
            "n_signal_total": len(sp.train) + len(sp.val) + len(sp.test),
            "n_train": len(sp.train), "n_validation": len(sp.val), "n_test": len(sp.test),
            "driver_first_valid_session": dt(sp.base_lo),
            "train_first_signal": dt(sp.train[0]) if sp.train else None,
            "train_last_signal_(window_end)": dt(sp.train_hi) if sp.train else None,
            "validation_window_start": dt(sp.val_lo) if sp.val else None,
            "validation_first_signal": dt(sp.val[0]) if sp.val else None,
            "validation_last_signal_(window_end)": dt(sp.val_hi) if sp.val else None,
            "test_window_start": dt(sp.test_lo) if sp.test else None,
            "test_first_signal": dt(sp.test[0]) if sp.test else None,
            "test_last_signal": dt(sp.test[-1]) if sp.test else None,
            "test_window_end": dt(td.n - 1) if sp.test else None,
            "ticker_first_session": dt(0), "ticker_last_session": dt(td.n - 1),
            "train_end_pct_of_ticker_calendar": (
                round((d[sp.train_hi] - d[0]).days / max(1, (d[-1] - d[0]).days) * 100, 1) if sp.train else None),
            "validation_end_pct_of_ticker_calendar": (
                round((d[sp.val_hi] - d[0]).days / max(1, (d[-1] - d[0]).days) * 100, 1) if sp.val else None),
        })
    return rows


def boundaries_summary(b):
    """Per ticker, for conditions with TRAIN n>=30 (the ones that can become candidates):
    how far apart the calendar cut dates are between conditions."""
    e = b[b["n_train"] >= rd.MIN_TRAIN_N].copy()
    rows = []
    for tk, g in e.groupby("ticker"):
        te = pd.to_datetime(g["train_last_signal_(window_end)"])
        ve = pd.to_datetime(g["validation_last_signal_(window_end)"])
        rows.append({
            "ticker": tk, "n_conditions_with_train_n30": len(g),
            "train_end_earliest": str(te.min().date()), "train_end_median": str(te.sort_values().iloc[len(te) // 2].date()),
            "train_end_latest": str(te.max().date()), "train_end_spread_days": int((te.max() - te.min()).days),
            "validation_end_earliest": str(ve.min().date()), "validation_end_latest": str(ve.max().date()),
            "validation_end_spread_days": int((ve.max() - ve.min()).days),
            "train_end_pct_of_calendar_min": g["train_end_pct_of_ticker_calendar"].min(),
            "train_end_pct_of_calendar_max": g["train_end_pct_of_ticker_calendar"].max(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CHECKPOINT CONVENTION SENSITIVITY (baseline only; real data; no conditions)
# ---------------------------------------------------------------------------
def checkpoint_sensitivity(td):
    rows = []
    for d in rd.DIRECTIONS:
        t = td.t_min[d]
        for cp in rd.PRIMARY_CHECKPOINTS:
            m = rd.CP_MINUTES[cp]
            for ti, tg in enumerate(rd.THRESHOLDS):
                if tg not in rd.TARGETS:
                    continue
                inc = float(np.mean((~np.isnan(t[:, ti])) & (t[:, ti] <= m)) * 100)
                strict = float(np.mean((~np.isnan(t[:, ti])) & (t[:, ti] < m)) * 100)
                at_edge = float(np.mean((~np.isnan(t[:, ti])) & (t[:, ti] == m)) * 100)
                rows.append({"ticker": td.ticker, "direction": d, "target_pct": tg, "checkpoint": cp,
                             "n_days": td.n, "baseline_hit_inclusive_pct": round(inc, 2),
                             "baseline_hit_strict_pct": round(strict, 2),
                             "difference_pp": round(inc - strict, 2),
                             "first_touch_in_boundary_bar_pct_of_days": round(at_edge, 2)})
    return rows


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------
def _fingerprint(driver_table, bundles, convention, min_shift):
    h = hashlib.sha1()
    h.update(f"{driver_table.shape}|{round(float(np.nansum(driver_table.to_numpy())), 4)}|{convention}|{min_shift}".encode())
    for tk, (td, _al) in sorted(bundles.items()):
        h.update(f"{tk}:{td.n}:{td.dates[0]}:{td.dates[-1]}:{int(td.hit['LONG'].sum())}".encode())
    return h.hexdigest()


def build_bundles(tickers, get_clean_days, driver_table, out_dir, progress=None, cache_hours=CACHE_HOURS):
    """Per ticker: outcome arrays (TickerData, original convention) + aligned driver block.
    TickerData is cached on disk so a resumed run does not repeat the slow engine pass."""
    os.makedirs(out_dir, exist_ok=True)
    bundles, errors = {}, []
    for i, tk in enumerate(tickers):
        if progress:
            progress("data", i, len(tickers), f"{tk}: loading sessions and outcomes")
        path = os.path.join(out_dir, f"td_{tk}.pkl")
        td = None
        if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < cache_hours * 3600:
            try:
                with open(path, "rb") as f:
                    td = pickle.load(f)
            except Exception:
                td = None
        try:
            if td is None:
                clean = get_clean_days(tk)
                if not clean:
                    errors.append((tk, "no clean sessions returned"))
                    continue
                td = rd.extract_ticker_data(tk, clean)
                del clean
                with open(path, "wb") as f:
                    pickle.dump(td, f)
            if td.n < 2 * MIN_SHIFT_SESSIONS + 50:
                errors.append((tk, f"only {td.n} usable sessions"))
                continue
            aligned = rd.align_drivers_for_ticker(driver_table, td.dates)
            bundles[tk] = (td, aligned)
        except Exception as e:
            errors.append((tk, f"{type(e).__name__}: {e}"))
    return bundles, errors


def _shift_row(tk, td, u, min_shift, real=False):
    k = 0 if real else shift_sessions(td.n, u, min_shift)
    return {"ticker": tk, "shift_sessions": k, "n_sessions": td.n,
            "approx_calendar_offset_days": int((td.dates[min(k, td.n - 1)] - td.dates[0]).days) if k else 0}


def run_one_pass(bundles, conditions, u, convention, min_shift, real=False):
    """One complete discovery/validation/test pass over all tickers. real=True -> unshifted."""
    rows, shifts = [], []
    for tk, (td, aligned) in bundles.items():
        tdc = apply_convention(td, convention)
        if real:
            al, k = aligned, 0
        else:
            k = shift_sessions(td.n, u, min_shift)
            al = roll_aligned(aligned, k)
        rows.extend(discover_validate_test(tdc, al, conditions))
        shifts.append(_shift_row(tk, td, u, min_shift, real))
    return pd.DataFrame(rows), shifts


def run_calibration(tickers, get_clean_days, driver_table, conditions, n_reps=20, seed=DEFAULT_SEED,
                    convention="original", out_dir="v3_placebo_out", progress=None, min_shift=MIN_SHIFT_SESSIONS):
    """Full calibration. Repetition 0 = real unshifted data; 1..n_reps = placebo."""
    if convention not in CONVENTIONS:
        raise ValueError(convention)
    conditions = [c for c in conditions if c[0] in set(driver_table.columns)]
    run_dir = os.path.join(out_dir, f"seed{seed}_{convention}")
    os.makedirs(run_dir, exist_ok=True)
    bundles, errors = build_bundles(tickers, get_clean_days, driver_table, out_dir, progress)
    if not bundles:
        raise RuntimeError(f"no ticker data could be built: {errors}")

    # resumability guard: saved reps are only reused if inputs are identical
    fp = _fingerprint(driver_table, bundles, convention, min_shift)
    fp_path = os.path.join(run_dir, "fingerprint.json")
    old = json.load(open(fp_path))["fp"] if os.path.exists(fp_path) else None
    if old != fp:
        for f in os.listdir(run_dir):
            if f.startswith("rep_") or f.endswith(".csv"):
                os.remove(os.path.join(run_dir, f))
        json.dump({"fp": fp}, open(fp_path, "w"))

    t_all = time.time()
    # ---- repetition 0: REAL ----
    real_path = os.path.join(run_dir, "rep_000_REAL.csv")
    if os.path.exists(real_path):
        real_combos = pd.read_csv(real_path)
    else:
        if progress:
            progress("rep", 0, n_reps + 1, "rep 0 (real, unshifted)")
        real_combos, _ = run_one_pass(bundles, conditions, 0.0, convention, min_shift, real=True)
        real_combos.to_csv(real_path, index=False)
    real_m = rep_metrics(real_combos)

    # ---- placebo repetitions ----
    rep_rows, shift_rows, rep_frames, rep_secs = [], [], {}, []
    for r in range(1, n_reps + 1):
        u = draw_u(seed, r)
        path = os.path.join(run_dir, f"rep_{r:03d}.csv")
        if progress:
            progress("rep", r, n_reps + 1, f"placebo repetition {r} of {n_reps}")
        t0 = time.time()
        if os.path.exists(path):
            combos = pd.read_csv(path)
            shifts = [_shift_row(tk, td, u, min_shift) for tk, (td, _a) in bundles.items()]
        else:
            combos, shifts = run_one_pass(bundles, conditions, u, convention, min_shift)
            combos.to_csv(path, index=False)
            rep_secs.append(time.time() - t0)
        rep_frames[r] = combos
        m = rep_metrics(combos)
        m.update({"rep": r, "u": round(u, 6)})
        rep_rows.append(m)
        for s in shifts:
            shift_rows.append({"rep": r, "u": round(u, 6), **s})
    placebo_df = pd.DataFrame(rep_rows)

    # ---- split boundaries (real run) + checkpoint sensitivity (real data, original td) ----
    b_rows, sens_rows = [], []
    for tk, (td, aligned) in bundles.items():
        splits = rd.build_signal_splits(aligned, conditions, td.n)
        b_rows.extend(boundaries_rows(td, splits))
        sens_rows.extend(checkpoint_sensitivity(td))
    boundaries = pd.DataFrame(b_rows)
    sens = pd.DataFrame(sens_rows)

    ref_match = {k: (real_m[k] == REAL_REFERENCE[k]) for k in ("n_candidates", "n_val_pass", "n_test_positive", "n_test_kept")}
    return {
        "errors": errors, "convention": convention, "seed": seed, "n_reps": n_reps, "min_shift": min_shift,
        "real_combos": real_combos, "real_metrics": real_m, "reference": REAL_REFERENCE,
        "reproduces_reference": bool(all(ref_match.values())) if convention == "original" else None,
        "reference_match_detail": ref_match,
        "placebo_metrics": placebo_df, "rep_combos": rep_frames,
        "summary": compare_to_placebo(real_m, placebo_df), "distribution": distribution_table(placebo_df),
        "shifts": pd.DataFrame(shift_rows), "boundaries": boundaries, "boundaries_summary": boundaries_summary(boundaries),
        "sensitivity": sens, "sensitivity_summary": sens.groupby(["checkpoint", "target_pct"])[
            ["baseline_hit_inclusive_pct", "baseline_hit_strict_pct", "difference_pp",
             "first_touch_in_boundary_bar_pct_of_days"]].mean().round(2).reset_index(),
        "seconds_per_new_rep": (float(np.mean(rep_secs)) if rep_secs else None),
        "seconds_total": time.time() - t_all,
    }


# ---------------------------------------------------------------------------
# AUDIT 1 -- DRIVER DAILY-BAR TIMING (AUD/USD and every other driver)
# ---------------------------------------------------------------------------
BOUNDARY_CANDIDATES = [
    ("UTC midnight", "UTC", 0),
    ("London midnight", "Europe/London", 0),
    ("New York midnight", "America/New_York", 0),
    ("New York 17:00", "America/New_York", 17),
    ("Sydney midnight", "Australia/Sydney", 0),
]


def _as_tz(idx, tz):
    idx = pd.DatetimeIndex(idx)
    return idx.tz_convert(tz) if idx.tz is not None else idx.tz_localize("UTC").tz_convert(tz)


def audit_one_driver(name, daily, hourly, tol=5e-4):
    """daily/hourly: yfinance-style frames (Open/High/Low/Close; hourly index tz-aware).
    Finds which day-boundary definition best reproduces the DAILY High/Low from the HOURLY
    bars, then reports when the information in a daily bar is actually complete in Sydney time,
    versus the 9:45am cutoff of the ASX session the pipeline pairs it with (next weekday)."""
    out = {"driver": name, "n_daily_rows": int(len(daily)), "n_hourly_rows": int(len(hourly))}
    if daily is None or hourly is None or len(daily) < 5 or len(hourly) < 24:
        out["result"] = "NOT ENOUGH DATA"
        return out
    d_idx = pd.DatetimeIndex(daily.index)
    out["daily_index_timezone"] = str(d_idx.tz) if d_idx.tz is not None else "naive"
    out["daily_index_time_of_day_sample"] = str(d_idx[-1].time())
    hr = hourly.copy()
    hr.index = pd.DatetimeIndex(hr.index)
    if hr.index.tz is None:
        hr.index = hr.index.tz_localize("UTC")
    best = None
    for label, tz, hour in BOUNDARY_CANDIDATES:
        hh = hr.copy()
        loc = hh.index.tz_convert(tz)
        shifted = loc - pd.Timedelta(hours=hour)       # boundary hour -> midnight
        hh["day"] = shifted.normalize().tz_localize(None)
        g = hh.groupby("day").agg(h=("High", "max"), l=("Low", "min"))
        dd = daily.copy()
        lab = d_idx.tz_localize(None) if d_idx.tz is not None else d_idx     # the date label as Yahoo prints it
        dd["day"] = lab.normalize()
        j = dd.join(g, on="day", how="inner")
        if len(j) < 5:
            continue
        ok = ((j["High"] - j["h"]).abs() / j["High"] < tol) & ((j["Low"] - j["l"]).abs() / j["Low"] < tol)
        rate = float(ok.mean())
        if best is None or rate > best[0]:
            best = (rate, label, tz, hour, len(j))
    if best is None:
        out["result"] = "NO BOUNDARY MATCH POSSIBLE"
        return out
    rate, label, tz, hour, nmatch = best
    out.update({"best_boundary": label, "boundary_match_rate": round(rate, 3), "days_compared": nmatch})
    if rate < 0.6:
        out["result"] = "UNRESOLVED (no boundary reproduces the daily High/Low well) - inspect raw evidence"
        return out
    # when is each daily bar's information complete? = end of the last hourly bar that day
    hh = hr.copy()
    loc = hh.index.tz_convert(tz)
    hh["day"] = (loc - pd.Timedelta(hours=hour)).normalize().tz_localize(None)
    last_end = hh.index.to_series().groupby(hh["day"]).max() + pd.Timedelta(hours=1)
    n_ok = n_tot = 0
    clocks = []
    for day, tend in last_end.items():
        syd = tend.tz_convert("Australia/Sydney")
        nxt = pd.Timestamp(day) + pd.offsets.BDay(1)            # the ASX session the pipeline pairs it with
        cutoff = pd.Timestamp(nxt.date(), tz="Australia/Sydney") + pd.Timedelta(hours=9, minutes=45)
        n_tot += 1
        n_ok += int(syd <= cutoff)
        clocks.append(syd.hour + syd.minute / 60)
    out.update({
        "last_bar_end_sydney_clock_min": round(min(clocks), 2), "last_bar_end_sydney_clock_max": round(max(clocks), 2),
        "pct_days_complete_before_0945_next_session": round(100 * n_ok / n_tot, 1), "days_checked": n_tot,
        "result": ("OK - always complete before 9:45" if n_ok == n_tot else
                   f"LOOK-AHEAD RISK - complete before 9:45 on only {100 * n_ok / n_tot:.0f}% of days"),
    })
    return out


def audit_driver_timestamps(drivers, fetch_daily, fetch_hourly, tickers_lookup=None):
    """drivers: list of driver names. fetch_*(yf_ticker) -> DataFrame or None (injectable for tests)."""
    from config_v1 import DRIVERS
    lookup = tickers_lookup or {n: DRIVERS[n][0] for n in drivers}
    rows, evidence = [], {}
    for n in drivers:
        tk = lookup[n]
        try:
            d, h = fetch_daily(tk), fetch_hourly(tk)
            r = audit_one_driver(n, d, h)
            r["yahoo_ticker"] = tk
        except Exception as e:
            r = {"driver": n, "yahoo_ticker": tk, "result": f"ERROR {type(e).__name__}: {e}"}
            d = h = None
        rows.append(r)
        if n == "audusd" and d is not None and h is not None:
            dd = d.tail(12).copy()
            dd.index = [str(x) for x in dd.index]
            hh = h.tail(60).copy()
            hh.index = [f"{x}  |  Sydney {pd.Timestamp(x).tz_convert('Australia/Sydney')}" for x in hh.index]
            evidence = {"audusd_daily_last12": dd.reset_index().rename(columns={"index": "daily_index_as_returned"}),
                        "audusd_hourly_last60": hh.reset_index().rename(columns={"index": "hourly_bar_start (UTC | Sydney)"})}
    return pd.DataFrame(rows), evidence


def make_yahoo_fetchers():
    import yfinance as yf

    def daily(tk):
        return yf.Ticker(tk).history(period="3mo", interval="1d", auto_adjust=False)

    def hourly(tk):
        return yf.Ticker(tk).history(period="3mo", interval="1h", auto_adjust=False)
    return daily, hourly


# ---------------------------------------------------------------------------
# AUDIT 2 -- EODHD 5-MINUTE BAR LABELLING (start vs end) AND OPEN/CLOSE AUCTION BARS
# ---------------------------------------------------------------------------
def audit_bar_labels(raw_bars):
    """raw_bars: list of EODHD 5m dicts. Looks at Sydney-local labels: which labels exist, and
    which label carries the opening / closing auction volume. Auction volume sits in the bar that
    CONTAINS 10:00 and ~16:10: if the biggest opening volume is labelled 10:00 and the biggest
    closing volume 16:10 the labels are bar START times; labelled 10:05 / 16:15 would mean END times."""
    from eodhd_logic import to_sydney_and_classify
    df, err = to_sydney_and_classify(raw_bars)
    if err:
        return {"error": err}, pd.DataFrame()
    df = df.copy()
    df["label"] = df["sydney_dt"].dt.strftime("%H:%M")
    per_label = df.groupby("label").agg(n_bars=("label", "size"), median_volume=("volume", "median"),
                                        n_days=("sydney_date", "nunique")).reset_index()
    first_win = per_label[(per_label["label"] >= "09:55") & (per_label["label"] <= "10:15")]
    last_win = per_label[(per_label["label"] >= "16:00") & (per_label["label"] <= "16:25")]
    open_peak = first_win.sort_values("median_volume", ascending=False).iloc[0]["label"] if len(first_win) else None
    close_peak = last_win.sort_values("median_volume", ascending=False).iloc[0]["label"] if len(last_win) else None
    days = df.groupby("sydney_date")["label"].agg(["min", "max"])
    cont = df[df["in_continuous_session"]].groupby("sydney_date").size()
    verdict = "UNCLEAR"
    if open_peak == "10:00":
        verdict = "labels are bar START times (opening-auction volume sits in the bar labelled 10:00)"
    elif open_peak == "10:05":
        verdict = "labels may be bar END times (opening-auction volume sits in the bar labelled 10:05) - REVIEW"
    findings = {
        "n_days": int(df["sydney_date"].nunique()), "n_bars": int(len(df)),
        "earliest_label_any_day": str(days["min"].min()), "latest_label_any_day": str(days["max"].max()),
        "label_with_biggest_opening_volume_(09:55-10:15)": open_peak,
        "label_with_biggest_closing_volume_(16:00-16:25)": close_peak,
        "median_continuous_bars_per_day": float(cont.median()) if len(cont) else None,
        "bars_labelled_exactly_16:00_present": bool((df["label"] == "16:00").any()),
        "verdict": verdict,
        "official_doc_statement": "EODHD documentation: timestamp = 'Bar start time as a Unix timestamp (UTC)'.",
        "engine_rule": "intraday_engine counts a bar toward a checkpoint if its START minute <= checkpoint minute, "
                       "so '10:15' includes the 10:15-10:20 bar (data up to 10:20), '10:30' up to 10:35, '11:00' up to 11:05.",
    }
    return findings, per_label


# ---------------------------------------------------------------------------
# README for the ZIP
# ---------------------------------------------------------------------------
README = """V3 PLACEBO CALIBRATION - what is in this ZIP and how to read it
=================================================================
Question: are the original reverse-discovery results (152 TRAIN candidates, 67 validation passes,
34 Test-kept rows) stronger than what randomised overnight data produces through the identical process?

METHOD
- Same 16 stocks, same 156 conditions (78 driver/threshold pairs x UP/DOWN), same 60/20/20 split, same
  TRAIN-only candidate selection, same validation rule, same Test check. Original code is imported, not edited.
- Placebo: each stock's aligned driver block (all drivers together) is rolled circularly by k ASX sessions.
  Preserves driver distributions, volatility clustering, driver-to-driver correlation, and the missing-value
  pattern; breaks the date link to the stock's morning moves. k >= 126 sessions (~6 months) from the true alignment
  both ways. Same relative shift for every stock within a repetition. Fixed seeds: repetition r is reproducible.
- Repetition 0 is the REAL unshifted data through the same code; it should reproduce 152 / 67 / 34 exactly
  (convention 'original').
- 'Test-kept' = passed validation AND Test uplift positive AND Test uplift >= 50% of TRAIN uplift.

FILES
- placebo_summary.csv           real vs placebo: mean, median, min, p05, p95, max, share of reps >= real, empirical p
- placebo_distribution.csv      how many repetitions produced each count
- placebo_rep_metrics.csv       one row per repetition (headline counts)
- rep_000_REAL.csv              the real run reproduced in this app (384 combination rows)
- rep_001.csv ... rep_NNN.csv   one file per placebo repetition (384 combination rows each)
- placebo_shifts.csv            the shift (sessions) used for each stock in each repetition
- split_boundaries_real.csv     calendar boundaries of TRAIN / VALIDATION / TEST for every ticker x condition
- split_boundaries_summary.csv  how those boundaries differ between conditions, per ticker
- checkpoint_sensitivity.csv    baseline hit rates with the engine's checkpoint rule vs a strict rule
- checkpoint_sensitivity_summary.csv

HOW SPLIT BOUNDARIES DIFFER BETWEEN CONDITIONS
Each condition's signal days are split 60/20/20 BY POSITION among that condition's own signal days. TRAIN therefore
ends at that condition's last TRAIN signal day, VALIDATION at its last validation signal day, and TEST runs to the end of
the stock's data. A condition whose signals are spread evenly gets cut points near 60% / 80% of the calendar; a condition
whose signals cluster in time gets cut points anywhere. So one condition's validation period can overlap another's
training period. This is inherent to the locked method; the files above list every boundary.

READING THE RESULT
- If the real value is above (or at the top of) the placebo range for the headline counts, the findings are stronger
  than chance on this data.
- 'share_equal_or_exceeding_real' is the fraction of placebo repetitions that matched or beat the real result.
  With 20 repetitions the smallest possible empirical p is 1/21 = 0.048; with 100 it is 1/101 = 0.0099.
- This test shows the findings exceed RANDOM date alignment. It does not by itself prove tradability, and it keeps
  any shared-volatility-regime effects as 'real' structure (a regime that makes both a driver and a stock volatile
  at the same time is real information, but is not a day-specific directional signal).
"""


# ---------------------------------------------------------------------------
# DRIVER DATA (cached so resumed runs use identical inputs), EODHD SAMPLE, ZIPS, VERDICT
# ---------------------------------------------------------------------------
def load_or_build_drivers(conditions, out_dir, cache_hours=CACHE_HOURS):
    """Same drivers, same builder as the original run. Cached for the session window so that a
    resumed run uses exactly the same driver data. A table with failed drivers is never cached."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "driver_table.pkl")
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < cache_hours * 3600:
        try:
            with open(path, "rb") as f:
                return pickle.load(f), []
        except Exception:
            pass
    table, failed = rd.build_drivers(conditions)
    if failed:
        time.sleep(5)
        names = {n for n, _ in failed}
        again, failed2 = rd.build_drivers([c for c in conditions if c[0] in names])
        if again is not None and not again.empty:
            table = table.join(again, how="outer")
        failed = failed2
    if not failed:
        with open(path, "wb") as f:
            pickle.dump(table, f)
    return table, failed


def fetch_eodhd_sample(token, symbol="CBA.AU", days=25):
    from datetime import datetime, timedelta, timezone
    from intraday_data import fetch_full_intraday_history
    end = datetime.now(timezone.utc)
    bars, failed = fetch_full_intraday_history(symbol, token, end - timedelta(days=days), end)
    if failed:
        raise RuntimeError(f"EODHD sample fetch failed: {failed[:1]}")
    return bars


def verdict_text(res):
    s = res["summary"].set_index("metric")
    R = res["n_reps"]
    lines = []
    for m in ("n_test_kept", "n_val_pass", "n_candidates"):
        r = s.loc[m]
        if r["reps_equal_or_exceeding_real"] == 0:
            lines.append(f"{METRIC_LABELS[m].split('  [')[0]}: real {r['real']:.0f} is higher than all {R} placebo repetitions "
                         f"(placebo mean {r['placebo_mean']}, range {r['placebo_min']:.0f} to {r['placebo_max']:.0f}).")
        else:
            lines.append(f"{METRIC_LABELS[m].split('  [')[0]}: real {r['real']:.0f}; {int(r['reps_equal_or_exceeding_real'])} of {R} placebo "
                         f"repetitions ({r['share_equal_or_exceeding_real']:.0%}) matched or beat it "
                         f"(placebo mean {r['placebo_mean']}, range {r['placebo_min']:.0f} to {r['placebo_max']:.0f}).")
    return lines


def build_results_zip(res):
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", README)
        z.writestr("placebo_summary.csv", res["summary"].to_csv(index=False))
        z.writestr("placebo_distribution.csv", res["distribution"].to_csv(index=False))
        z.writestr("placebo_rep_metrics.csv", res["placebo_metrics"].to_csv(index=False))
        z.writestr("rep_000_REAL.csv", res["real_combos"].to_csv(index=False))
        for r, df in sorted(res["rep_combos"].items()):
            z.writestr(f"rep_{r:03d}.csv", df.to_csv(index=False))
        z.writestr("placebo_shifts.csv", res["shifts"].to_csv(index=False))
        z.writestr("split_boundaries_real.csv", res["boundaries"].to_csv(index=False))
        z.writestr("split_boundaries_summary.csv", res["boundaries_summary"].to_csv(index=False))
        z.writestr("checkpoint_sensitivity.csv", res["sensitivity"].to_csv(index=False))
        z.writestr("checkpoint_sensitivity_summary.csv", res["sensitivity_summary"].to_csv(index=False))
        ref = pd.DataFrame([{"metric": k, "reference_original_run": v, "reproduced_in_this_app": res["real_metrics"].get(k)}
                            for k, v in res["reference"].items()])
        z.writestr("real_vs_reference.csv", ref.to_csv(index=False))
        z.writestr("run_settings.json", json.dumps({
            "seed": res["seed"], "n_reps": res["n_reps"], "convention": res["convention"], "min_shift_sessions": res["min_shift"],
            "reproduces_original_run": res["reproduces_reference"], "ticker_errors": res["errors"],
            "seconds_total": round(res["seconds_total"], 1)}, indent=2, default=str))
    return buf.getvalue()


def build_audit_zip(driver_table, evidence, findings, label_table):
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("audit_driver_timing.csv", driver_table.to_csv(index=False))
        for k, df in (evidence or {}).items():
            z.writestr(f"audit_{k}.csv", df.to_csv(index=False))
        z.writestr("audit_eodhd_bar_findings.json", json.dumps(findings, indent=2, default=str))
        if label_table is not None and len(label_table):
            z.writestr("audit_eodhd_bar_labels.csv", label_table.to_csv(index=False))
    return buf.getvalue()
