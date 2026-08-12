"""
BACKTEST ENGINE
==================
Pure statistics on top of the tables historical_data.py builds. No
invented scores anywhere here — every number is computed from real
matched historical sessions, or the function returns "insufficient
sample" rather than a number.
"""

import math
import numpy as np
import pandas as pd

from config_v1 import (
    GAP_BUCKETS, COSTS, SAMPLE_SIZE_BANDS, SPLIT_RATIOS,
)


def sample_band(n):
    for lo, hi, label, note in SAMPLE_SIZE_BANDS:
        if lo <= n < hi:
            return label, note
    return "insufficient", "Not shown as an edge — informational only"


def wilson_ci(win_rate, n, z=1.96):
    """Wilson score interval on a win rate — honest uncertainty rather
    than a bare point estimate, especially important at small N."""
    if n == 0:
        return (float("nan"), float("nan"))
    denom = 1 + z**2 / n
    centre = win_rate + z**2 / (2 * n)
    adj = z * math.sqrt((win_rate * (1 - win_rate) + z**2 / (4 * n)) / n)
    lo = (centre - adj) / denom
    hi = (centre + adj) / denom
    return (max(0.0, lo), min(1.0, hi))


def apply_costs(returns, costs=None):
    """Subtracts round-trip commission + slippage (in bps) from each
    return. Slippage defaults to 0 and is reported as unset, not
    modelled — never silently baked into a number as if it were real."""
    costs = costs or COSTS
    total_bps = costs["commission_bps_roundtrip"] + costs["slippage_bps_roundtrip"]
    return returns - (total_bps / 100.0)


def compute_stats(returns, costs=None):
    """Full stats bundle for a series of % returns. Returns None (with
    a reason) if the sample is empty."""
    returns = pd.Series(returns).dropna()
    n = len(returns)
    if n == 0:
        return {"n": 0, "band": "insufficient", "note": "No matching historical sessions"}

    winners = returns[returns > 0]
    losers = returns[returns < 0]

    win_rate = len(winners) / n
    ci_lo, ci_hi = wilson_ci(win_rate, n)

    mean_ret = returns.mean()
    median_ret = returns.median()
    avg_winner = winners.mean() if len(winners) else float("nan")
    median_winner = winners.median() if len(winners) else float("nan")
    avg_loser = losers.mean() if len(losers) else float("nan")
    median_loser = losers.median() if len(losers) else float("nan")

    payoff_ratio = (avg_winner / abs(avg_loser)) if (len(winners) and len(losers) and avg_loser != 0) else float("nan")
    gross_win = winners.sum()
    gross_loss = abs(losers.sum())
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("nan")

    expectancy = win_rate * (avg_winner if len(winners) else 0) + (1 - win_rate) * (avg_loser if len(losers) else 0)
    stdev = returns.std()

    band, band_note = sample_band(n)

    net_returns = apply_costs(returns, costs)
    net_expectancy = net_returns.mean()

    return {
        "n": n,
        "band": band,
        "band_note": band_note,
        "win_rate": round(win_rate * 100, 1),
        "win_rate_ci": (round(ci_lo * 100, 1), round(ci_hi * 100, 1)),
        "mean_return": round(mean_ret, 3),
        "median_return": round(median_ret, 3),
        "avg_winner": round(avg_winner, 3) if not math.isnan(avg_winner) else None,
        "median_winner": round(median_winner, 3) if not math.isnan(median_winner) else None,
        "avg_loser": round(avg_loser, 3) if not math.isnan(avg_loser) else None,
        "median_loser": round(median_loser, 3) if not math.isnan(median_loser) else None,
        "payoff_ratio": round(payoff_ratio, 2) if not math.isnan(payoff_ratio) else None,
        "profit_factor": round(profit_factor, 2) if not math.isnan(profit_factor) else None,
        "expectancy_gross": round(expectancy, 3),
        "expectancy_net": round(net_expectancy, 3),
        "stdev": round(stdev, 3),
    }


def gap_bucket_for(gap_pct, direction):
    """Direction-adjusted gap bucket: a SHORT setup's 'favourable' gap
    is a negative one, so we bucket the gap in the direction of the
    trade, not its raw sign."""
    if gap_pct is None or (isinstance(gap_pct, float) and math.isnan(gap_pct)):
        return None
    signed = gap_pct if direction == "LONG" else -gap_pct
    for label, lo, hi in GAP_BUCKETS:
        if lo <= signed < hi:
            return label
    return None


def chronological_split(dates, ratios=None):
    """Splits a sorted sequence of dates into train/validation/test by
    POSITION, not randomly — critical for time series, prevents future
    information leaking into training."""
    ratios = ratios or SPLIT_RATIOS
    dates = sorted(dates)
    n = len(dates)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["validation"])
    train = dates[:n_train]
    val = dates[n_train:n_train + n_val]
    test = dates[n_train + n_val:]
    return train, val, test


def evaluate_hypothesis(hypothesis, aligned_driver_table, asx_outcomes_by_ticker,
                         asx_theme_stocks, costs=None):
    """Evaluates one hypothesis across its ASX basket, split
    chronologically into train/validation/test, computed separately —
    never optimised against test, test is touched exactly once here.

    Returns a dict keyed by outcome column (open_to_close, next_session,
    day2, day3), each containing train/validation/test stats plus a
    gap-bucket breakdown computed on validation only (test stays
    untouched until final reporting).
    """
    theme = hypothesis["theme"]
    direction = hypothesis["direction"]
    usable_from = pd.Timestamp(hypothesis["usable_from"])
    tickers = asx_theme_stocks[theme]

    driver_slice = aligned_driver_table[aligned_driver_table.index >= usable_from]
    matched_dates = [d for d in driver_slice.index if hypothesis["condition"](driver_slice.loc[d])]

    if not matched_dates:
        return {"hypothesis_id": hypothesis["id"], "n_matched": 0, "note": "No historical matches found"}

    train_dates, val_dates, test_dates = chronological_split(matched_dates)

    outcome_columns = ["open_to_close_pct", "next_session_return", "day2_return", "day3_return"]
    result = {"hypothesis_id": hypothesis["id"], "label": hypothesis["label"],
              "direction": direction, "theme": theme, "n_matched": len(matched_dates),
              "n_train": len(train_dates), "n_validation": len(val_dates), "n_test": len(test_dates)}

    sign = 1 if direction == "LONG" else -1

    for col in outcome_columns:
        split_results = {}
        for split_name, split_dates in [("train", train_dates), ("validation", val_dates), ("test", test_dates)]:
            rows = []
            for d in split_dates:
                for t in tickers:
                    df = asx_outcomes_by_ticker.get(t)
                    if df is None or d not in df.index:
                        continue
                    val = df.loc[d, col]
                    if pd.isna(val):
                        continue
                    rows.append(val * sign)
            split_results[split_name] = compute_stats(rows, costs)
        result[col] = split_results

    # Gap-bucket breakdown, validation set only (test stays untouched)
    gap_breakdown = {}
    for d in val_dates:
        for t in tickers:
            df = asx_outcomes_by_ticker.get(t)
            if df is None or d not in df.index:
                continue
            gap = df.loc[d, "gap_pct"]
            ret = df.loc[d, "open_to_close_pct"]
            if pd.isna(gap) or pd.isna(ret):
                continue
            bucket = gap_bucket_for(gap, direction)
            if bucket is None:
                continue
            gap_breakdown.setdefault(bucket, []).append(ret * sign)

    result["gap_breakdown_validation"] = {
        bucket: compute_stats(rets, costs) for bucket, rets in gap_breakdown.items()
    }

    return result
