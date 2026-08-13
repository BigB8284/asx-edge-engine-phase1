"""
INTRADAY STATS — aggregation, baseline comparison, plain-English summaries
================================================================================
Takes the per-day outputs of intraday_engine.compute_day_outcomes and
aggregates them into signal-day stats, baseline (unconditional) stats,
and the comparison between the two. Every threshold (1/2/3/5%) is
reported on equal footing — none is treated as the headline.
"""

import math
from intraday_engine import THRESHOLDS, CHECKPOINTS, MFE_MAE_WINDOWS

try:
    from backtest import wilson_ci  # reuse the already-tested implementation
except ImportError:
    def wilson_ci(win_rate, n, z=1.96):
        if n == 0:
            return (float("nan"), float("nan"))
        p = win_rate / 100 if win_rate > 1 else win_rate
        denom = 1 + z**2 / n
        centre = p + z**2 / (2 * n)
        adj = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
        return (max(0.0, (centre - adj) / denom), min(1.0, (centre + adj) / denom))


def median(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    n = len(v)
    mid = n // 2
    return v[mid] if n % 2 else (v[mid - 1] + v[mid]) / 2


def aggregate_outcomes(day_outcomes_list):
    """day_outcomes_list: list of dicts from compute_day_outcomes (None
    entries, from excluded/incomplete days, must already be filtered out
    before calling this)."""
    days = [d for d in day_outcomes_list if d is not None]
    n_days = len(days)
    result = {"n_days": n_days}

    # --- Threshold-level: probability reached, CI, time-to-threshold, MAE-before ---
    threshold_stats = {}
    for t in THRESHOLDS:
        reached_flags = [d["thresholds"][t]["reached"] for d in days]
        n_reached = sum(reached_flags)
        prob = n_reached / n_days if n_days else 0
        ci_lo, ci_hi = wilson_ci(prob, n_days)
        times = [d["thresholds"][t]["time_minutes"] for d in days if d["thresholds"][t]["reached"]]
        maes_before = [d["thresholds"][t]["mae_before"] for d in days if d["thresholds"][t]["reached"]]
        threshold_stats[t] = {
            "n_days": n_days, "n_reached": n_reached,
            "probability": round(prob * 100, 1), "ci_lo": round(ci_lo * 100, 1), "ci_hi": round(ci_hi * 100, 1),
            "median_time_to_threshold_min": median(times),
            "median_mae_before_reached": round(median(maes_before), 3) if maes_before else None,
        }
    result["thresholds"] = threshold_stats

    # --- Threshold x checkpoint: probability reached BY each checkpoint ---
    checkpoint_stats = {}
    for cp in CHECKPOINTS:
        checkpoint_stats[cp] = {}
        for t in THRESHOLDS:
            flags = [d["reached_by_checkpoint"][cp][t] for d in days]
            n_reached = sum(flags)
            prob = n_reached / n_days if n_days else 0
            ci_lo, ci_hi = wilson_ci(prob, n_days)
            checkpoint_stats[cp][t] = {"n_days": n_days, "n_reached": n_reached,
                                        "probability": round(prob * 100, 1),
                                        "ci_lo": round(ci_lo * 100, 1), "ci_hi": round(ci_hi * 100, 1)}
    result["checkpoints"] = checkpoint_stats

    # --- MFE/MAE distributions per window ---
    window_stats = {}
    for w in MFE_MAE_WINDOWS:
        mfes = [d["windows"][w]["mfe"] for d in days if d["windows"][w]["mfe"] is not None]
        maes = [d["windows"][w]["mae"] for d in days if d["windows"][w]["mae"] is not None]
        ttm = [d["windows"][w]["time_to_mfe_minutes"] for d in days if d["windows"][w]["time_to_mfe_minutes"] is not None]
        window_stats[w] = {
            "median_mfe": round(median(mfes), 3) if mfes else None,
            "median_mae": round(median(maes), 3) if maes else None,
            "median_time_to_mfe_min": median(ttm),
        }
    result["windows"] = window_stats

    return result


def compute_baseline_delta(signal_agg, baseline_agg):
    """Compares signal-day stats against baseline (unconditional) stats.
    Reports BOTH values plus the delta for every threshold and
    checkpoint — never collapses to a single number."""
    comparison = {"checkpoints": {}}
    for cp in CHECKPOINTS:
        comparison["checkpoints"][cp] = {}
        for t in THRESHOLDS:
            sig = signal_agg["checkpoints"][cp][t]
            base = baseline_agg["checkpoints"][cp][t]
            comparison["checkpoints"][cp][t] = {
                "signal_probability": sig["probability"], "signal_n": sig["n_days"],
                "signal_ci": (sig["ci_lo"], sig["ci_hi"]),
                "baseline_probability": base["probability"], "baseline_n": base["n_days"],
                "baseline_ci": (base["ci_lo"], base["ci_hi"]),
                "delta_pp": round(sig["probability"] - base["probability"], 1),
            }
    return comparison


def format_summary_line(ticker, direction, signal_agg, baseline_agg, delta):
    """Produces the exact plain-English style requested:
    'PDN LONG — signal days: 38% reached +2% by 11:00 vs 17% normally; ...'
    """
    parts = [f"{ticker} {direction} — signal days:"]
    for t in [2.0, 3.0]:  # the two most illustrative thresholds for the headline line; full detail carries all four
        d = delta["checkpoints"]["11:00"][t]
        parts.append(f"{d['signal_probability']:.0f}% reached +{t:.0f}% by 11:00 vs {d['baseline_probability']:.0f}% normally;")
    mfe = signal_agg["windows"]["full_session"]["median_mfe"]
    ttm = signal_agg["windows"]["full_session"]["median_time_to_mfe_min"]
    mae_before_2pct = signal_agg["thresholds"][2.0]["median_mae_before_reached"]
    parts.append(f"median MFE {'+' if mfe and mfe>=0 else ''}{mfe:.1f}%;" if mfe is not None else "median MFE n/a;")
    if ttm is not None:
        h, m = divmod(int(ttm), 60)
        parts.append(f"median time-to-MFE {h}:{m:02d};")
    if mae_before_2pct is not None:
        parts.append(f"median MAE before +2% = {mae_before_2pct:.1f}%.")
    return " ".join(parts)
