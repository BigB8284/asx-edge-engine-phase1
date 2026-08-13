"""
INTRADAY ENGINE — path-dependent threshold/MFE/MAE scoring
================================================================
Core logic for V3. Walks a day's 5-minute bars chronologically and
scores threshold-reaching, MFE/MAE, and timing the way a trader
actually experiences a session — not by comparing two static prices.

All outcomes are computed relative to the day's OPEN and signed by
direction (LONG: favourable = higher; SHORT: favourable = lower), so
the same functions work for both without duplicating logic.

Documented, unavoidable limitation: 5-minute bars don't reveal whether
a bar's high or low came first. Every function here uses a fixed,
conservative convention — within each bar, the ADVERSE extreme is
processed before the FAVOURABLE extreme — so MAE-before-threshold is
never understated. This is a stated assumption, not a hidden one.
"""

import pandas as pd
import numpy as np
import math

THRESHOLDS = [1.0, 2.0, 3.0, 5.0]
CHECKPOINTS = ["10:15", "10:30", "11:00", "12:00", "full_session"]
MFE_MAE_WINDOWS = [30, 60, 90, "full_session"]


def signed_pct(price, open_price, direction):
    """% move from open, signed so positive always means 'in the
    signal's favour' regardless of LONG/SHORT."""
    sign = 1 if direction == "LONG" else -1
    return (price - open_price) / open_price * 100 * sign


def compute_day_outcomes(bars, direction):
    """bars: DataFrame of ONE day's continuous-session 5-minute bars,
    already sorted chronologically, with columns open/high/low/close
    and a 'minutes_from_open' column (0 for the first bar).
    direction: 'LONG' or 'SHORT'.

    Returns a dict of outcomes for this single day. Bars must already
    be filtered to a single COMPLETE trading day before calling this —
    this function does not check completeness itself.
    """
    if bars.empty:
        return None

    open_price = bars.iloc[0]["open"]
    result = {"open_price": open_price, "n_bars": len(bars)}

    # Favourable/adverse extreme per bar, direction-aware, computed once.
    if direction == "LONG":
        bars = bars.assign(
            favourable_extreme=lambda d: signed_pct(d["high"], open_price, direction),
            adverse_extreme=lambda d: signed_pct(d["low"], open_price, direction),
        )
    else:
        bars = bars.assign(
            favourable_extreme=lambda d: signed_pct(d["low"], open_price, direction),
            adverse_extreme=lambda d: signed_pct(d["high"], open_price, direction),
        )

    # --- Threshold crossing: time-to-threshold, MAE-before-threshold ---
    running_mfe = -math.inf
    running_mae = math.inf
    threshold_results = {t: {"reached": False, "time_minutes": None, "mae_before": None} for t in THRESHOLDS}

    for _, bar in bars.iterrows():
        # Conservative convention: process the ADVERSE extreme first.
        running_mae = min(running_mae, bar["adverse_extreme"])
        # Then the favourable extreme, checking for new threshold crossings.
        for t in THRESHOLDS:
            if not threshold_results[t]["reached"] and bar["favourable_extreme"] >= t:
                threshold_results[t]["reached"] = True
                threshold_results[t]["time_minutes"] = bar["minutes_from_open"]
                threshold_results[t]["mae_before"] = running_mae if running_mae != math.inf else 0.0
        running_mfe = max(running_mfe, bar["favourable_extreme"])

    result["thresholds"] = threshold_results

    # --- reached_by_checkpoint: independent re-check restricted to bars up to each checkpoint ---
    checkpoint_minutes = {"10:15": 15, "10:30": 30, "11:00": 60, "12:00": 120, "full_session": math.inf}
    reached_by_checkpoint = {}
    for cp_name, cp_minutes in checkpoint_minutes.items():
        sub = bars[bars["minutes_from_open"] <= cp_minutes]
        reached_by_checkpoint[cp_name] = {
            t: bool((sub["favourable_extreme"] >= t).any()) if not sub.empty else False
            for t in THRESHOLDS
        }
    result["reached_by_checkpoint"] = reached_by_checkpoint

    # --- MFE/MAE per window, with time-to-MFE ---
    window_results = {}
    for w in MFE_MAE_WINDOWS:
        w_minutes = w if isinstance(w, (int, float)) else math.inf
        sub = bars[bars["minutes_from_open"] <= w_minutes]
        if sub.empty:
            window_results[w] = {"mfe": None, "mae": None, "time_to_mfe_minutes": None}
            continue
        mfe = sub["favourable_extreme"].max()
        mae = sub["adverse_extreme"].min()
        mfe_row = sub[sub["favourable_extreme"] == mfe].iloc[0]
        window_results[w] = {"mfe": round(mfe, 4), "mae": round(mae, 4),
                             "time_to_mfe_minutes": int(mfe_row["minutes_from_open"])}
    result["windows"] = window_results

    return result
