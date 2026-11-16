"""
V3 REVERSE DISCOVERY — core logic (no Streamlit in this file)
================================================================
Implements the LOCKED reverse-discovery specification (approved
2026-08-25, saved in the Trading project as v3-reverse-discovery-spec.md).

Question answered:
  For each of the 16 pilot stocks, which single pre-open driver
  condition (from the pre-registered 78-condition list) best separates
  morning-mover days from ordinary days on TRAIN data -- and does that
  relationship survive ONE validation check and ONE test check?

Pipeline per ticker:
  1. Own clean 5-minute history  -> per-day outcomes from the REAL
     intraday_engine.compute_day_outcomes (not reimplemented).
  2. Drivers aligned with the REAL historical_data.align_to_asx_sessions
     (backward-only merge_asof, allow_exact_matches=False) -> every
     condition is knowable before the ASX open. No look-ahead.
  3. Each condition (driver, threshold, side UP/DOWN) is its own signal.
     Signal days are split 60/20/20 by POSITION using the REAL
     backtest.chronological_split, independently per ticker/condition.
  4. TRAIN ONLY: for every (direction, target%, primary checkpoint) the
     signal hit-rate is compared with that ticker's own unconditional
     baseline hit-rate over the same TRAIN window. A signal is eligible
     if n_train >= 30 AND uplift >= +10pp.
  5. Exactly ONE candidate (best TRAIN uplift) per (ticker, direction,
     target%, checkpoint) is FROZEN. The discovery functions only ever
     receive TRAIN-sliced arrays, so validation/test cannot leak in.
  6. VALIDATION (once): same direction, positive uplift, and keeps at
     least 50% of the TRAIN uplift. No minimum validation n (it is always
     reported). Failure is final: no candidate #2.
  7. TEST (once, validation-passers only): reported, never used to tune.

Primary checkpoints (discovery/gating): 10:15, 10:30, 11:00.
Secondary checkpoints (11:30, 12:00, full_session): descriptive only,
computed AFTER the freeze, never used for selection.

Nothing here is an opportunity filter. Every ticker x direction x
condition x target x checkpoint cell is exported (ungated matrix) and
every combination that did not become a candidate is logged with a reason.
"""

import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backtest import chronological_split, wilson_ci, gap_bucket_for
from config_v1 import DRIVERS
from driver_threshold_sanity_check import build_all_conditions
from historical_data import build_driver_table, align_to_asx_sessions
import intraday_engine as _eng
from intraday_engine import compute_day_outcomes, THRESHOLDS, CHECKPOINTS, MFE_MAE_WINDOWS

# --- fail loudly if an older engine (pre-V3) is deployed -------------------
if 4.0 not in THRESHOLDS or "11:30" not in CHECKPOINTS:
    raise ImportError(
        "intraday_engine.py is the pre-V3 version (needs THRESHOLDS incl. 4.0 and "
        "CHECKPOINT '11:30'). Use the V3-edited engine from the repo."
    )

try:  # the locked roster lives in v3_pilot_config; fall back to the spec list
    from v3_pilot_config import PILOT_TICKERS
except Exception:  # pragma: no cover - only when run outside the repo
    PILOT_TICKERS = [
        "CIA.AX", "ILU.AX", "ARU.AX", "NXT.AX", "GMG.AX", "BPT.AX", "SFR.AX", "WDS.AX",
        "BOE.AX", "CBA.AX", "QBE.AX", "S32.AX", "JHX.AX", "NEM.AX", "BHP.AX", "RIO.AX",
    ]

# ---------------------------------------------------------------------------
# LOCKED PARAMETERS (spec 2026-08-25) -- do not edit to chase results
# ---------------------------------------------------------------------------
TARGETS = [1.0, 2.0, 3.0, 4.0]
PRIMARY_CHECKPOINTS = ["10:15", "10:30", "11:00"]
SECONDARY_CHECKPOINTS = ["11:30", "12:00", "full_session"]
MIN_TRAIN_N = 30
MIN_TRAIN_UPLIFT_PP = 10.0
VALIDATION_RETENTION = 0.5          # validation uplift must be >= 50% of TRAIN uplift
DIRECTIONS = ["LONG", "SHORT"]
SIDES = ["UP", "DOWN"]              # UP: driver >= +threshold ; DOWN: driver <= -threshold
SWEET_SPOT_EXPONENT = 0.3           # frozen scoring formula (informational ranking only)
SPEED_BONUS = {"10:15": 1.05, "10:30": 1.03, "11:00": 1.01}
CP_MINUTES = {"10:15": 15, "10:30": 30, "11:00": 60, "11:30": 90, "12:00": 120, "full_session": math.inf}

_CP_IDX = {cp: i for i, cp in enumerate(CHECKPOINTS)}
_T_IDX = {t: i for i, t in enumerate(THRESHOLDS)}
_W_IDX = {w: i for i, w in enumerate(MFE_MAE_WINDOWS)}
_P_CP = [_CP_IDX[c] for c in PRIMARY_CHECKPOINTS]
_P_T = [_T_IDX[t] for t in TARGETS]


def tier_for(uplift_pp):
    if uplift_pp is None or (isinstance(uplift_pp, float) and math.isnan(uplift_pp)):
        return "n/a"
    if uplift_pp >= 20:
        return "very_strong_20pp+"
    if uplift_pp >= 15:
        return "strong_15-20pp"
    if uplift_pp >= MIN_TRAIN_UPLIFT_PP:
        return "modest_10-15pp"
    return "below_gate"


def sweet_spot_score(prob_pct, target_pct, checkpoint):
    """Frozen project formula: probability x target^0.3 x speed bonus.
    Informational ranking only -- never a gate."""
    if prob_pct is None or (isinstance(prob_pct, float) and math.isnan(prob_pct)):
        return float("nan")
    return round(prob_pct * (target_pct ** SWEET_SPOT_EXPONENT) * SPEED_BONUS.get(checkpoint, 1.0), 2)


# ---------------------------------------------------------------------------
# PER-TICKER OUTCOME ARRAYS
# ---------------------------------------------------------------------------
@dataclass
class TickerData:
    ticker: str
    dates: list                      # list[pd.Timestamp], one per usable session, ascending
    gap_pct: np.ndarray              # open vs prior close (retrospective description only)
    hit: dict = field(default_factory=dict)       # dir -> (n, n_cp, n_t) bool
    t_min: dict = field(default_factory=dict)     # dir -> (n, n_t) float, minutes to first touch
    mae_before: dict = field(default_factory=dict)  # dir -> (n, n_t) float
    win_mfe: dict = field(default_factory=dict)   # dir -> (n, n_w)
    win_mae: dict = field(default_factory=dict)   # dir -> (n, n_w)

    @property
    def n(self):
        return len(self.dates)


def extract_ticker_data(ticker, clean_days):
    """clean_days: {date_str: bars_df} from intraday_data.build_clean_day_groups.
    Runs the REAL engine for LONG and SHORT on every clean day."""
    dates, gaps = [], []
    per_dir = {d: {"hit": [], "t": [], "mae": [], "mfe_w": [], "mae_w": []} for d in DIRECTIONS}
    prev_close, prev_date = None, None
    for date_str in sorted(clean_days):
        bars = clean_days[date_str]
        outs = {d: compute_day_outcomes(bars, d) for d in DIRECTIONS}
        valid = bars.dropna(subset=["open", "high", "low", "close"])
        if any(o is None for o in outs.values()) or valid.empty:
            continue
        ts = pd.Timestamp(date_str)
        open_price = outs["LONG"]["open_price"]
        if prev_close is not None and (ts - prev_date).days <= 6 and prev_close:
            gaps.append((open_price - prev_close) / prev_close * 100)
        else:
            gaps.append(np.nan)
        prev_close, prev_date = float(valid.iloc[-1]["close"]), ts
        dates.append(ts)
        for d in DIRECTIONS:
            o = outs[d]
            per_dir[d]["hit"].append([[bool(o["reached_by_checkpoint"][cp][t]) for t in THRESHOLDS]
                                      for cp in CHECKPOINTS])
            per_dir[d]["t"].append([o["thresholds"][t]["time_minutes"] if o["thresholds"][t]["reached"] else np.nan
                                    for t in THRESHOLDS])
            per_dir[d]["mae"].append([o["thresholds"][t]["mae_before"] if o["thresholds"][t]["reached"] else np.nan
                                      for t in THRESHOLDS])
            per_dir[d]["mfe_w"].append([o["windows"][w]["mfe"] if o["windows"][w]["mfe"] is not None else np.nan
                                        for w in MFE_MAE_WINDOWS])
            per_dir[d]["mae_w"].append([o["windows"][w]["mae"] if o["windows"][w]["mae"] is not None else np.nan
                                        for w in MFE_MAE_WINDOWS])
    td = TickerData(ticker, dates, np.array(gaps, dtype=float))
    for d in DIRECTIONS:
        td.hit[d] = np.array(per_dir[d]["hit"], dtype=bool).reshape(len(dates), len(CHECKPOINTS), len(THRESHOLDS))
        td.t_min[d] = np.array(per_dir[d]["t"], dtype=float).reshape(len(dates), len(THRESHOLDS))
        td.mae_before[d] = np.array(per_dir[d]["mae"], dtype=float).reshape(len(dates), len(THRESHOLDS))
        td.win_mfe[d] = np.array(per_dir[d]["mfe_w"], dtype=float).reshape(len(dates), len(MFE_MAE_WINDOWS))
        td.win_mae[d] = np.array(per_dir[d]["mae_w"], dtype=float).reshape(len(dates), len(MFE_MAE_WINDOWS))
    return td


# ---------------------------------------------------------------------------
# SIGNALS AND SPLITS (driver side only -- no outcomes touched here)
# ---------------------------------------------------------------------------
@dataclass
class SignalSplit:
    key: tuple                # (driver, threshold, side)
    train: list               # day indices (positions in TickerData)
    val: list
    test: list
    base_lo: int              # first day the driver was usable (baseline window start)
    train_hi: int             # last TRAIN signal day (baseline window end for TRAIN)
    val_lo: int
    val_hi: int
    test_lo: int
    test_hi: int

    def window(self, split):
        return {"train": (self.base_lo, self.train_hi),
                "validation": (self.val_lo, self.val_hi),
                "test": (self.test_lo, self.test_hi)}[split]

    def idx(self, split):
        return {"train": self.train, "validation": self.val, "test": self.test}[split]


def align_drivers_for_ticker(driver_table, dates, drivers_lookup=None):
    """Real backward-only alignment of every driver to this ticker's own
    session dates, with each driver nulled before its own first_available date."""
    drivers_lookup = drivers_lookup if drivers_lookup is not None else DRIVERS
    aligned = align_to_asx_sessions(driver_table, dates)
    for name in aligned.columns:
        usable_from = pd.Timestamp(drivers_lookup[name][2])
        aligned.loc[aligned.index < usable_from, name] = np.nan
    return aligned


def build_signal_splits(aligned, conditions, n_days):
    """One SignalSplit per (driver, threshold, side). Position-based
    60/20/20 split of that condition's own signal days."""
    splits = {}
    for name, th, _src in conditions:
        if name not in aligned.columns:
            continue
        col = aligned[name].to_numpy(dtype=float)
        valid = ~np.isnan(col)
        if not valid.any():
            continue
        base_lo = int(np.argmax(valid))
        for side in SIDES:
            with np.errstate(invalid="ignore"):
                mask = (col >= th) if side == "UP" else (col <= -th)
            idx = [int(i) for i in np.flatnonzero(mask & valid)]
            tr, va, te = chronological_split(idx) if idx else ([], [], [])
            train_hi = tr[-1] if tr else base_lo - 1
            val_lo = train_hi + 1
            val_hi = va[-1] if va else train_hi
            test_lo = val_hi + 1
            splits[(name, th, side)] = SignalSplit(
                (name, th, side), tr, va, te, base_lo, train_hi, val_lo, val_hi, test_lo, n_days - 1)
    return splits


# ---------------------------------------------------------------------------
# TRAIN-ONLY DISCOVERY -- these functions receive ONLY train-sliced arrays
# ---------------------------------------------------------------------------
def train_stats(sig_hits_train, base_hits_train):
    """sig_hits_train: (n_sig, 3 primary cps, 4 targets) bool, TRAIN signal days only.
    base_hits_train: (n_base, 3, 4) bool, all days in the TRAIN window.
    Returns per-cell signal prob / baseline prob / uplift (pp)."""
    n_sig, n_base = len(sig_hits_train), len(base_hits_train)
    nan = np.full((len(PRIMARY_CHECKPOINTS), len(TARGETS)), np.nan)
    if n_sig == 0 or n_base == 0:
        return {"n_sig": n_sig, "n_base": n_base, "sig": nan, "base": nan, "uplift": nan}
    sig = sig_hits_train.mean(axis=0) * 100
    base = base_hits_train.mean(axis=0) * 100
    return {"n_sig": n_sig, "n_base": n_base, "sig": sig, "base": base, "uplift": sig - base}


def train_view_stats(H, sp):
    """The ONLY place discovery touches outcome data. Slices H (n, cp, t)
    to TRAIN signal days and the TRAIN baseline window (which ends at the
    last TRAIN signal day), then hands just those slices to train_stats."""
    assert sp.base_lo <= sp.train_hi and sp.train_hi == sp.train[-1]
    sig = H[np.array(sp.train)][:, _P_CP][:, :, _P_T]
    base = H[sp.base_lo:sp.train_hi + 1][:, _P_CP][:, :, _P_T]
    return train_stats(sig, base)


def validation_verdict(train_uplift_pp, val_uplift_pp, n_val):
    """Locked validation rule. Returns (passed, reason, retention)."""
    if n_val == 0 or val_uplift_pp is None or math.isnan(val_uplift_pp):
        return False, "no validation signal days (n=0)", None
    retention = val_uplift_pp / train_uplift_pp
    if val_uplift_pp <= 0:
        return False, "validation uplift not positive (direction not preserved)", retention
    if retention < VALIDATION_RETENTION:
        return False, (f"validation kept {retention*100:.0f}% of TRAIN uplift "
                       f"(<{VALIDATION_RETENTION*100:.0f}%)"), retention
    return True, "validation passed", retention


def select_candidate(per_signal_train_stats):
    """per_signal_train_stats: {key: train_stats-dict}. For each (cp, target)
    returns the single best eligible signal key (or None) plus the best
    n>=30 near-miss for the failure log. Ties: larger n, then key order."""
    out = {}
    for ci, cp in enumerate(PRIMARY_CHECKPOINTS):
        for ti, t in enumerate(TARGETS):
            elig, near = [], []
            for key, st in per_signal_train_stats.items():
                if st["n_sig"] < MIN_TRAIN_N or math.isnan(st["uplift"][ci, ti]):
                    continue
                u = float(st["uplift"][ci, ti])
                near.append((u, st["n_sig"], key))
                if u >= MIN_TRAIN_UPLIFT_PP:
                    elig.append((u, st["n_sig"], key))
            best = None
            if elig:
                best = sorted(elig, key=lambda x: (-x[0], -x[1], str(x[2])))[0][2]
            nm = sorted(near, key=lambda x: (-x[0], -x[1], str(x[2])))[0] if near else None
            out[(cp, t)] = {"candidate": best, "n_eligible": len(elig), "n_with_n30": len(near), "near_miss": nm}
    return out


# ---------------------------------------------------------------------------
# DESCRIPTIVE HELPERS (applied to frozen candidates only)
# ---------------------------------------------------------------------------
def _rate(h):
    return float(np.mean(h) * 100) if len(h) else float("nan")


def _med(a):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    return float(np.median(a)) if len(a) else float("nan")


def _r(x, nd=2):
    return None if (x is None or (isinstance(x, float) and math.isnan(x))) else round(float(x), nd)


def _split_block(td, direction, sp, split, cp, t):
    """Signal vs baseline stats for ONE split of ONE frozen candidate."""
    ci, ti = _CP_IDX[cp], _T_IDX[t]
    idx = np.array(sp.idx(split), dtype=int)
    lo, hi = sp.window(split)
    base_hits = td.hit[direction][lo:hi + 1, ci, ti] if hi >= lo else np.array([], dtype=bool)
    sig_hits = td.hit[direction][idx, ci, ti] if len(idx) else np.array([], dtype=bool)
    sig_p, base_p = _rate(sig_hits), _rate(base_hits)
    ci_lo, ci_hi = wilson_ci(sig_p / 100, len(sig_hits)) if len(sig_hits) else (float("nan"), float("nan"))
    hit_idx = idx[sig_hits] if len(idx) else idx
    return {
        "n_signal": len(idx), "n_baseline": int(max(0, hi - lo + 1)),
        "signal_prob": sig_p, "baseline_prob": base_p,
        "uplift_pp": sig_p - base_p if not (math.isnan(sig_p) or math.isnan(base_p)) else float("nan"),
        "ci_lo": ci_lo * 100, "ci_hi": ci_hi * 100,
        "median_time_to_target_min": _med(td.t_min[direction][hit_idx, ti]) if len(hit_idx) else float("nan"),
        "median_mae_before_target": _med(td.mae_before[direction][hit_idx, ti]) if len(hit_idx) else float("nan"),
    }


def _continuation_rows(td, direction, sp, split, t, meta):
    rows = []
    for cp in SECONDARY_CHECKPOINTS:
        b = _split_block(td, direction, sp, split, cp, t)
        rows.append({**meta, "split": split, "kind": "continuation", "checkpoint": cp,
                     "window_min": None, "n_signal": b["n_signal"], "n_baseline": b["n_baseline"],
                     "signal_prob": _r(b["signal_prob"], 1), "baseline_prob": _r(b["baseline_prob"], 1),
                     "uplift_pp": _r(b["uplift_pp"], 1), "median_mfe": None, "median_mae": None})
    idx = np.array(sp.idx(split), dtype=int)
    lo, hi = sp.window(split)
    for w in MFE_MAE_WINDOWS:
        wi = _W_IDX[w]
        rows.append({**meta, "split": split, "kind": "mfe_mae_signal_days", "checkpoint": None,
                     "window_min": w, "n_signal": len(idx), "n_baseline": None, "signal_prob": None,
                     "baseline_prob": None, "uplift_pp": None,
                     "median_mfe": _r(_med(td.win_mfe[direction][idx, wi]) if len(idx) else float("nan"), 3),
                     "median_mae": _r(_med(td.win_mae[direction][idx, wi]) if len(idx) else float("nan"), 3)})
        rows.append({**meta, "split": split, "kind": "mfe_mae_baseline_days", "checkpoint": None,
                     "window_min": w, "n_signal": None, "n_baseline": int(max(0, hi - lo + 1)),
                     "signal_prob": None, "baseline_prob": None, "uplift_pp": None,
                     "median_mfe": _r(_med(td.win_mfe[direction][lo:hi + 1, wi]) if hi >= lo else float("nan"), 3),
                     "median_mae": _r(_med(td.win_mae[direction][lo:hi + 1, wi]) if hi >= lo else float("nan"), 3)})
    return rows


def _gap_rows(td, direction, sp, split, cp, t, meta):
    ci, ti = _CP_IDX[cp], _T_IDX[t]
    idx = np.array(sp.idx(split), dtype=int)
    buckets = {}
    for i in idx:
        b = gap_bucket_for(float(td.gap_pct[i]), direction)
        buckets.setdefault(b or "gap_unavailable", []).append(i)
    rows = []
    for b, ii in buckets.items():
        ii = np.array(ii, dtype=int)
        rows.append({**meta, "split": split, "checkpoint": cp, "gap_bucket": b, "n_days": len(ii),
                     "hit_rate": _r(_rate(td.hit[direction][ii, ci, ti]), 1)})
    return rows


# ---------------------------------------------------------------------------
# ONE TICKER, END TO END
# ---------------------------------------------------------------------------
def process_ticker(td, aligned, conditions):
    """Returns dict of row-lists: matrix, combos, continuation, gap, plus meta."""
    splits = build_signal_splits(aligned, conditions, td.n)
    matrix, combos, cont, gaps = [], [], [], []

    for direction in DIRECTIONS:
        H = td.hit[direction]
        # ---- 1. TRAIN-only discovery: slice arrays BEFORE calling discovery ----
        stats = {}
        for key, sp in splits.items():
            if len(sp.train) == 0:
                stats[key] = train_stats(np.zeros((0, 3, 4), bool), np.zeros((0, 3, 4), bool))
                continue
            stats[key] = train_view_stats(H, sp)

        # ---- ungated matrix (TRAIN stats only for every cell) ----
        for key, sp in splits.items():
            st = stats[key]
            for ci, cp in enumerate(PRIMARY_CHECKPOINTS):
                for ti, t in enumerate(TARGETS):
                    u = st["uplift"][ci, ti] if st["n_sig"] else float("nan")
                    matrix.append({
                        "ticker": td.ticker, "direction": direction, "driver": key[0],
                        "driver_threshold_pct": key[1], "driver_side": key[2],
                        "target_pct": t, "checkpoint": cp,
                        "n_train": len(sp.train), "n_validation": len(sp.val), "n_test": len(sp.test),
                        "n_train_baseline_days": st["n_base"],
                        "train_signal_prob": _r(st["sig"][ci, ti] if st["n_sig"] else float("nan"), 1),
                        "train_baseline_prob": _r(st["base"][ci, ti] if st["n_sig"] else float("nan"), 1),
                        "train_uplift_pp": _r(u, 1), "tier": tier_for(u) if st["n_sig"] else "n/a",
                        "clears_train_gate": bool(st["n_sig"] >= MIN_TRAIN_N and not math.isnan(u)
                                                  and u >= MIN_TRAIN_UPLIFT_PP),
                    })

        # ---- 2. freeze one candidate per (cp, target) ----
        chosen = select_candidate(stats)

        # ---- 3/4. validation once, test once ----
        for cp in PRIMARY_CHECKPOINTS:
            for t in TARGETS:
                ch = chosen[(cp, t)]
                row = {"ticker": td.ticker, "direction": direction, "target_pct": t, "checkpoint": cp,
                       "n_signals_eligible_on_train": ch["n_eligible"],
                       "n_signals_with_train_n30": ch["n_with_n30"]}
                key = ch["candidate"]
                if key is None:
                    nm = ch["near_miss"]
                    row.update({
                        "status": "NOT_A_CANDIDATE",
                        "reason": f"no condition with TRAIN n>={MIN_TRAIN_N} and uplift>=+{MIN_TRAIN_UPLIFT_PP:.0f}pp",
                        "best_near_miss_condition": (f"{nm[2][0]} {nm[2][2]} {nm[2][1]:g}%" if nm else None),
                        "best_near_miss_train_uplift_pp": _r(nm[0], 1) if nm else None,
                        "best_near_miss_train_n": nm[1] if nm else None,
                    })
                    combos.append(row)
                    continue

                sp = splits[key]
                ci, ti = _CP_IDX[cp], _T_IDX[t]
                tr = _split_block(td, direction, sp, "train", cp, t)
                va = _split_block(td, direction, sp, "validation", cp, t)
                row.update({
                    "driver": key[0], "driver_threshold_pct": key[1], "driver_side": key[2],
                    "n_train": tr["n_signal"], "n_validation": va["n_signal"], "n_test": len(sp.test),
                    "train_signal_prob": _r(tr["signal_prob"], 1), "train_baseline_prob": _r(tr["baseline_prob"], 1),
                    "train_uplift_pp": _r(tr["uplift_pp"], 1), "train_tier": tier_for(tr["uplift_pp"]),
                    "train_median_time_to_target_min": _r(tr["median_time_to_target_min"], 0),
                    "train_median_mae_before_target": _r(tr["median_mae_before_target"], 3),
                    "val_signal_prob": _r(va["signal_prob"], 1), "val_baseline_prob": _r(va["baseline_prob"], 1),
                    "val_uplift_pp": _r(va["uplift_pp"], 1),
                    "val_median_time_to_target_min": _r(va["median_time_to_target_min"], 0),
                    "val_median_mae_before_target": _r(va["median_mae_before_target"], 3),
                    "score_train": sweet_spot_score(tr["signal_prob"], t, cp),
                })
                meta = {"ticker": td.ticker, "direction": direction, "target_pct": t, "checkpoint_primary": cp,
                        "driver": key[0], "driver_threshold_pct": key[1], "driver_side": key[2]}

                # validation verdict
                passed, reason, retention = validation_verdict(tr["uplift_pp"], va["uplift_pp"], va["n_signal"])
                row["val_retention_of_train"] = _r(retention, 2)
                row["validation_passed"] = passed
                row["reason"] = reason
                row["status"] = "VALIDATION_PASSED" if passed else "FAILED_VALIDATION"

                splits_used = ["train", "validation"]
                if passed:
                    te = _split_block(td, direction, sp, "test", cp, t)
                    splits_used.append("test")
                    tret = (te["uplift_pp"] / tr["uplift_pp"]) if not math.isnan(te["uplift_pp"]) else None
                    row.update({
                        "n_test": te["n_signal"], "test_signal_prob": _r(te["signal_prob"], 1),
                        "test_baseline_prob": _r(te["baseline_prob"], 1), "test_uplift_pp": _r(te["uplift_pp"], 1),
                        "test_ci_lo": _r(te["ci_lo"], 1), "test_ci_hi": _r(te["ci_hi"], 1),
                        "test_retention_of_train": _r(tret, 2),
                        # informational only -- same yardstick as validation, never used to change anything
                        "test_same_direction_positive": bool(not math.isnan(te["uplift_pp"]) and te["uplift_pp"] > 0),
                        "test_retains_50pct_of_train": bool(tret is not None and tret >= VALIDATION_RETENTION),
                        "test_median_time_to_target_min": _r(te["median_time_to_target_min"], 0),
                        "test_median_mae_before_target": _r(te["median_mae_before_target"], 3),
                        "score_test": sweet_spot_score(te["signal_prob"], t, cp),
                    })
                combos.append(row)

                for split in splits_used:
                    cont.extend(_continuation_rows(td, direction, sp, split, t, meta))
                    gaps.extend(_gap_rows(td, direction, sp, split, cp, t, meta))

    n_cond_signals = len(splits)
    meta = {"ticker": td.ticker, "n_sessions": td.n,
            "first_session": str(td.dates[0].date()) if td.dates else None,
            "last_session": str(td.dates[-1].date()) if td.dates else None,
            "n_signal_definitions": n_cond_signals}
    return {"matrix": matrix, "combos": combos, "continuation": cont, "gap": gaps, "meta": meta}


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------
def build_drivers(conditions=None, fetch_fn=None):
    """Fetch every driver used by the 78 conditions once. Returns
    (driver_table, failed). Failed drivers are REPORTED, never silently dropped."""
    conditions = conditions or build_all_conditions()
    names = sorted({c[0] for c in conditions})
    table, failed = build_driver_table(names, fetch_fn=fetch_fn, drivers_lookup=DRIVERS)
    return table, failed


def run_all(tickers, get_clean_days, driver_table, conditions=None, progress=None, drivers_lookup=None):
    """get_clean_days(ticker) -> {date_str: bars_df}. Bars are released after
    each ticker's outcome arrays are extracted (memory-safe)."""
    conditions = conditions or build_all_conditions()
    present = set(driver_table.columns)
    usable_conditions = [c for c in conditions if c[0] in present]
    dropped = sorted({c[0] for c in conditions if c[0] not in present})
    out = {"matrix": [], "combos": [], "continuation": [], "gap": [], "meta": [], "errors": [],
           "dropped_drivers": dropped, "n_conditions_used": len(usable_conditions)}
    for i, ticker in enumerate(tickers):
        if progress:
            progress(i, len(tickers), f"{ticker}: loading clean sessions")
        try:
            clean_days = get_clean_days(ticker)
            if not clean_days:
                out["errors"].append((ticker, "no clean sessions returned"))
                continue
            td = extract_ticker_data(ticker, clean_days)
            del clean_days
            if td.n < 100:
                out["errors"].append((ticker, f"only {td.n} usable sessions"))
                continue
            if progress:
                progress(i, len(tickers), f"{ticker}: {td.n} sessions, searching drivers (TRAIN only)")
            aligned = align_drivers_for_ticker(driver_table, td.dates, drivers_lookup)
            res = process_ticker(td, aligned, usable_conditions)
            for k in ("matrix", "combos", "continuation", "gap"):
                out[k].extend(res[k])
            out["meta"].append(res["meta"])
        except Exception as e:  # one bad ticker must not kill the run
            out["errors"].append((ticker, f"{type(e).__name__}: {e}"))
    if progress:
        progress(len(tickers), len(tickers), "done")
    return out


def results_to_frames(out):
    frames = {k: pd.DataFrame(out[k]) for k in ("matrix", "combos", "continuation", "gap", "meta")}
    c = frames["combos"]
    if not c.empty:
        order = {"VALIDATION_PASSED": 0, "FAILED_VALIDATION": 1, "NOT_A_CANDIDATE": 2}
        c["_o"] = c["status"].map(order)
        st_ = c["score_test"] if "score_test" in c else pd.Series(np.nan, index=c.index)
        c["_s"] = st_.fillna(c["score_train"] if "score_train" in c else 0)
        c = c.sort_values(["_o", "_s"], ascending=[True, False]).drop(columns=["_o", "_s"]).reset_index(drop=True)
        frames["combos"] = c
    return frames


# ---------------------------------------------------------------------------
# DATA ACCESS (EODHD) -- thin wrapper over the REAL pipeline
# ---------------------------------------------------------------------------
INTRADAY_FETCH_FLOOR = datetime(2020, 10, 1, tzinfo=timezone.utc)   # a floor only: each ticker's
# real start is whatever its own clean sessions begin at (WDS ~2022-05, NEM ~2023-10)


def eodhd_symbol(ticker):
    return ticker.replace(".AX", ".AU")


def make_eodhd_clean_days_getter(api_token, cache_dir="v3_rd_cache", max_cache_age_hours=20):
    from intraday_data import fetch_full_intraday_history, build_clean_day_groups
    os.makedirs(cache_dir, exist_ok=True)

    def get(ticker):
        path = os.path.join(cache_dir, f"{ticker}_raw5m.parquet")
        raw = None
        if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < max_cache_age_hours * 3600:
            try:
                raw = pd.read_parquet(path).to_dict("records")
            except Exception:
                raw = None
        if raw is None:
            raw, failed = fetch_full_intraday_history(eodhd_symbol(ticker), api_token,
                                                      INTRADAY_FETCH_FLOOR, datetime.now(timezone.utc))
            if failed:
                raise RuntimeError(f"EODHD windows failed for {ticker}: {failed[:2]}")
            if raw:
                try:
                    pd.DataFrame(raw).to_parquet(path)
                except Exception:
                    pass
        clean_days, _excl, _flag, _out = build_clean_day_groups(raw)
        del raw
        return clean_days

    return get
