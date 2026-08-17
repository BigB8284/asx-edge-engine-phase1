"""
INTRADAY STATS — aggregation, baseline comparison, plain-English summaries
================================================================================
Takes the per-day outputs of intraday_engine.compute_day_outcomes and
aggregates them into signal-day stats, baseline (unconditional) stats,
and the comparison between the two. Every threshold (now 1/2/3/4% per
the V3 spec, was 1/2/3/5%) is reported on equal footing — none is
treated as the headline.

V3 ADDITION (2026-08-14): compute_target_before_adverse(), below. This
answers "how often does the target get reached before a given adverse
move" — e.g. "+2% reached before -1%". It requires NO changes to
intraday_engine.py's core loop: each day's `mae_before` field, captured
at the exact moment a threshold is first reached, already reflects the
engine's existing conservative same-bar convention (adverse extreme
processed before favourable extreme within a bar). This function is a
pure aggregation over that already-correct per-day data.
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


DEFAULT_ADVERSE_LEVELS_PCT = [-0.5, -1.0, -2.0]


def compute_target_before_adverse(day_outcomes_list, adverse_levels_pct=None):
    """For each threshold T, computes P(T reached AND the worst adverse
    excursion before reaching it stayed better than A) for each adverse
    level A in adverse_levels_pct. Denominator is ALL days (n_days),
    matching how the existing threshold "probability" stat is computed
    — this is "how often does this specific, better outcome happen",
    not "of the days it worked, how clean was the path".

    Relies entirely on the per-day `mae_before` field already computed
    by intraday_engine.compute_day_outcomes under its existing
    conservative same-bar convention. No new per-bar logic here.

    day_outcomes_list: list of dicts from compute_day_outcomes (None
    entries must already be filtered out, same convention as
    aggregate_outcomes above).

    Returns: {threshold: {adverse_level: {"probability": pct, "n_days": n,
              "n_reached_before_adverse": count}}}
    """
    if adverse_levels_pct is None:
        adverse_levels_pct = DEFAULT_ADVERSE_LEVELS_PCT

    days = [d for d in day_outcomes_list if d is not None]
    n_days = len(days)
    result = {}

    for t in THRESHOLDS:
        result[t] = {}
        for a in adverse_levels_pct:
            n_reached_before_adverse = 0
            for d in days:
                th = d["thresholds"][t]
                if th["reached"] and th["mae_before"] is not None and th["mae_before"] > a:
                    n_reached_before_adverse += 1
            prob = n_reached_before_adverse / n_days if n_days else 0
            result[t][a] = {
                "probability": round(prob * 100, 1),
                "n_days": n_days,
                "n_reached_before_adverse": n_reached_before_adverse,
            }
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
    for t in [2.0, 3.0]:
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
