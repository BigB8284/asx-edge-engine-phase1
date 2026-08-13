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


def _collect_stock_day_returns(dates, tickers, asx_outcomes_by_ticker, col, sign):
    """Every (date, stock) pair as its own observation — the existing
    pooled granularity."""
    rows = []
    for d in dates:
        for t in tickers:
            df = asx_outcomes_by_ticker.get(t)
            if df is None or d not in df.index:
                continue
            val = df.loc[d, col]
            if pd.isna(val):
                continue
            rows.append(val * sign)
    return rows


def _collect_day_level_returns(dates, tickers, asx_outcomes_by_ticker, col, sign):
    """One observation PER DAY: the basket's average return across
    whichever stocks had valid data that day. This is the correct
    input for a genuine day-count win rate/CI, since stock-day pooling
    treats correlated same-day moves across a basket as independent
    evidence, which overstates confidence."""
    rows = []
    for d in dates:
        day_vals = []
        for t in tickers:
            df = asx_outcomes_by_ticker.get(t)
            if df is None or d not in df.index:
                continue
            val = df.loc[d, col]
            if pd.isna(val):
                continue
            day_vals.append(val * sign)
        if day_vals:
            rows.append(sum(day_vals) / len(day_vals))
    return rows


def _collect_per_stock_returns(dates, ticker, asx_outcomes_by_ticker, col, sign):
    """Returns for ONE stock only, across the matched dates — reveals
    whether a basket-level result is broad or concentrated in one name."""
    df = asx_outcomes_by_ticker.get(ticker)
    if df is None:
        return []
    rows = []
    for d in dates:
        if d not in df.index:
            continue
        val = df.loc[d, col]
        if pd.isna(val):
            continue
        rows.append(val * sign)
    return rows


def evaluate_hypothesis_detailed(hypothesis, aligned_driver_table, asx_outcomes_by_ticker,
                                  asx_theme_stocks, costs=None):
    """Same matched-date logic and same chronological split as
    evaluate_hypothesis — this does not change what counts as a match,
    it reports the SAME result at three granularities:
      - stock_day: pooled (date, stock) pairs, as before
      - day_level: one observation per day (basket average), the
        correct basis for a day-count win rate/CI
      - per_stock: each basket member's own stats, to check breadth
    """
    theme = hypothesis["theme"]
    direction = hypothesis["direction"]
    usable_from = pd.Timestamp(hypothesis["usable_from"])
    tickers = asx_theme_stocks[theme]

    driver_slice = aligned_driver_table[aligned_driver_table.index >= usable_from]
    matched_dates = [d for d in driver_slice.index if hypothesis["condition"](driver_slice.loc[d])]

    if not matched_dates:
        return {"hypothesis_id": hypothesis["id"], "n_matched_days": 0, "note": "No historical matches found"}

    train_dates, val_dates, test_dates = chronological_split(matched_dates)
    outcome_columns = ["open_to_close_pct", "next_session_return", "day2_return", "day3_return"]
    sign = 1 if direction == "LONG" else -1

    result = {"hypothesis_id": hypothesis["id"], "label": hypothesis["label"],
              "direction": direction, "theme": theme, "tickers": tickers,
              "n_matched_days": len(matched_dates),
              "n_train_days": len(train_dates), "n_validation_days": len(val_dates), "n_test_days": len(test_dates)}

    splits = [("train", train_dates), ("validation", val_dates), ("test", test_dates)]

    for col in outcome_columns:
        result.setdefault("stock_day", {})[col] = {
            split_name: compute_stats(_collect_stock_day_returns(split_dates, tickers, asx_outcomes_by_ticker, col, sign), costs)
            for split_name, split_dates in splits
        }
        result.setdefault("day_level", {})[col] = {
            split_name: compute_stats(_collect_day_level_returns(split_dates, tickers, asx_outcomes_by_ticker, col, sign), costs)
            for split_name, split_dates in splits
        }
        per_stock = {}
        for t in tickers:
            per_stock[t] = {
                split_name: compute_stats(_collect_per_stock_returns(split_dates, t, asx_outcomes_by_ticker, col, sign), costs)
                for split_name, split_dates in splits
            }
        result.setdefault("per_stock", {})[col] = per_stock

    return result


def compare_confirmed_vs_unconfirmed_by_split(base_hypothesis, confirmed_hypothesis, aligned_driver_table,
                                               asx_outcomes_by_ticker, asx_theme_stocks, costs=None):
    """Same confirmed-vs-unconfirmed-only isolation as compare_confirmed_vs_unconfirmed,
    but split chronologically (train/validation/test, computed independently
    on each date group since they're different date sets) AND broken out
    per ticker as well as basket-pooled. This is the out-of-sample version
    of that comparison — the pooled-history version answers "did this
    exist in the data we've looked at," this answers "does it hold up
    out of sample and is it broad or concentrated."
    """
    theme = base_hypothesis["theme"]
    direction = base_hypothesis["direction"]
    usable_from = pd.Timestamp(base_hypothesis["usable_from"])
    tickers = asx_theme_stocks[theme]
    sign = 1 if direction == "LONG" else -1

    driver_slice = aligned_driver_table[aligned_driver_table.index >= usable_from]
    base_dates = set(d for d in driver_slice.index if base_hypothesis["condition"](driver_slice.loc[d]))
    confirmed_dates = set(d for d in driver_slice.index if confirmed_hypothesis["condition"](driver_slice.loc[d]))

    if not confirmed_dates.issubset(base_dates):
        return {"error": "confirmed_hypothesis condition is not a subset of base_hypothesis condition"}

    unconfirmed_only_dates = sorted(base_dates - confirmed_dates)
    confirmed_dates = sorted(confirmed_dates)

    outcome_columns = ["open_to_close_pct", "next_session_return", "day2_return", "day3_return"]
    result = {"base_id": base_hypothesis["id"], "confirmed_id": confirmed_hypothesis["id"], "tickers": tickers}

    for group_name, group_dates in [("confirmed", confirmed_dates), ("unconfirmed_only", unconfirmed_only_dates)]:
        train_d, val_d, test_d = chronological_split(group_dates)
        splits = [("train", train_d), ("validation", val_d), ("test", test_d)]
        result[group_name] = {
            "n_total_days": len(group_dates),
            "n_train_days": len(train_d), "n_validation_days": len(val_d), "n_test_days": len(test_d),
        }
        for col in outcome_columns:
            result[group_name].setdefault("day_level", {})[col] = {
                split_name: compute_stats(_collect_day_level_returns(split_dates, tickers, asx_outcomes_by_ticker, col, sign), costs)
                for split_name, split_dates in splits
            }
            per_stock = {}
            for t in tickers:
                per_stock[t] = {
                    split_name: compute_stats(_collect_per_stock_returns(split_dates, t, asx_outcomes_by_ticker, col, sign), costs)
                    for split_name, split_dates in splits
                }
            result[group_name].setdefault("per_stock", {})[col] = per_stock

    return result


def gap_relationship_analysis(hypothesis, aligned_driver_table, asx_outcomes_by_ticker,
                               asx_theme_stocks, costs=None):
    """Does the overnight edge survive, strengthen, or disappear depending
    on how much the ASX stock has already gapped by the open? Diagnostic
    only — this reports pre-specified gap buckets (from config_v1.GAP_BUCKETS,
    not newly invented here) AND a monotonic correlation summary. It does
    NOT search for or select a threshold; any eventual threshold decision
    is a separate step, made on train+validation and checked once on test.

    Diagnostic set = train+validation dates POOLED (enough sample to see
    a pattern). Test dates are analysed separately and reported
    separately, touched once, not used to pick anything.
    """
    theme = hypothesis["theme"]
    direction = hypothesis["direction"]
    usable_from = pd.Timestamp(hypothesis["usable_from"])
    tickers = asx_theme_stocks[theme]
    sign = 1 if direction == "LONG" else -1

    driver_slice = aligned_driver_table[aligned_driver_table.index >= usable_from]
    matched_dates = [d for d in driver_slice.index if hypothesis["condition"](driver_slice.loc[d])]
    if not matched_dates:
        return {"hypothesis_id": hypothesis["id"], "n_matched_days": 0, "note": "No historical matches found"}

    train_d, val_d, test_d = chronological_split(matched_dates)
    diagnostic_dates = sorted(train_d + val_d)  # train+validation pooled

    def bucket_and_correlate(dates, label):
        # Bucketed view (basket day-level AND per-ticker), using the
        # EXISTING gap buckets from config_v1 — not new thresholds.
        bucket_result = {}
        rows_by_bucket_daylevel = {}
        rows_by_bucket_per_ticker = {t: {} for t in tickers}
        # Correlation inputs: one (signed_gap, signed_return) pair per (date, ticker) — per-ticker level,
        # plus a day-level version using basket-average gap and basket-average return.
        per_ticker_gap, per_ticker_ret = {t: [] for t in tickers}, {t: [] for t in tickers}
        day_level_gap, day_level_ret = [], []

        for d in dates:
            day_gaps, day_rets = [], []
            for t in tickers:
                df = asx_outcomes_by_ticker.get(t)
                if df is None or d not in df.index:
                    continue
                gap = df.loc[d, "gap_pct"]
                ret = df.loc[d, "open_to_close_pct"]
                if pd.isna(gap) or pd.isna(ret):
                    continue
                signed_gap = gap * sign
                signed_ret = ret * sign
                per_ticker_gap[t].append(signed_gap)
                per_ticker_ret[t].append(signed_ret)
                day_gaps.append(signed_gap)
                day_rets.append(signed_ret)
                bucket = gap_bucket_for(gap, direction)
                if bucket:
                    rows_by_bucket_per_ticker[t].setdefault(bucket, []).append(signed_ret)
            if day_gaps:
                avg_gap = sum(day_gaps) / len(day_gaps)
                avg_ret = sum(day_rets) / len(day_rets)
                day_level_gap.append(avg_gap)
                day_level_ret.append(avg_ret)
                # bucket the day-level average gap using the same buckets
                bucket = None
                for blabel, lo, hi in GAP_BUCKETS:
                    if lo <= avg_gap < hi:
                        bucket = blabel
                        break
                if bucket:
                    rows_by_bucket_daylevel.setdefault(bucket, []).append(avg_ret)

        bucket_stats_daylevel = {b: compute_stats(vals, costs) for b, vals in rows_by_bucket_daylevel.items()}
        bucket_stats_per_ticker = {
            t: {b: compute_stats(vals, costs) for b, vals in buckets.items()}
            for t, buckets in rows_by_bucket_per_ticker.items()
        }
        corr_day_level = (pd.Series(day_level_gap).corr(pd.Series(day_level_ret), method="spearman")
                           if len(day_level_gap) >= 5 else None)
        corr_per_ticker = {
            t: (pd.Series(per_ticker_gap[t]).corr(pd.Series(per_ticker_ret[t]), method="spearman")
                if len(per_ticker_gap[t]) >= 5 else None)
            for t in tickers
        }
        return {
            "label": label, "n_days": len(dates),
            "gap_buckets_day_level": bucket_stats_daylevel,
            "gap_buckets_per_ticker": bucket_stats_per_ticker,
            "spearman_gap_vs_return_day_level": corr_day_level,
            "spearman_gap_vs_return_per_ticker": corr_per_ticker,
        }

    return {
        "hypothesis_id": hypothesis["id"], "label": hypothesis["label"], "direction": direction, "theme": theme,
        "n_matched_days": len(matched_dates),
        "diagnostic_train_validation": bucket_and_correlate(diagnostic_dates, "train+validation (diagnostic)"),
        "held_out_test": bucket_and_correlate(test_d, "test (checked once)"),
    }


def compare_confirmed_vs_unconfirmed(base_hypothesis, confirmed_hypothesis, aligned_driver_table,
                                      asx_outcomes_by_ticker, asx_theme_stocks, costs=None):
    """Isolates what a confirmation driver actually added, using MATCHED
    DATES rather than comparing headline percentages from two different
    samples. base_hypothesis's condition must be implied by
    confirmed_hypothesis's condition (e.g. H4 <- H4b) so that
    confirmed's matched dates are a subset of base's.

    Returns stats for:
      - 'confirmed': days where BOTH conditions held (already reported
        elsewhere, included here for direct side-by-side comparison)
      - 'unconfirmed_only': days where the base condition held but the
        extra confirming condition did NOT — the true counterfactual
        for "did the confirmation add anything"
    """
    theme = base_hypothesis["theme"]
    direction = base_hypothesis["direction"]
    usable_from = pd.Timestamp(base_hypothesis["usable_from"])
    tickers = asx_theme_stocks[theme]
    sign = 1 if direction == "LONG" else -1

    driver_slice = aligned_driver_table[aligned_driver_table.index >= usable_from]
    base_dates = set(d for d in driver_slice.index if base_hypothesis["condition"](driver_slice.loc[d]))
    confirmed_dates = set(d for d in driver_slice.index if confirmed_hypothesis["condition"](driver_slice.loc[d]))

    if not confirmed_dates.issubset(base_dates):
        return {"error": "confirmed_hypothesis condition is not a subset of base_hypothesis condition — "
                          "not a valid confirmed-vs-unconfirmed comparison for this pair"}

    unconfirmed_only_dates = sorted(base_dates - confirmed_dates)
    confirmed_dates = sorted(confirmed_dates)

    outcome_columns = ["open_to_close_pct", "next_session_return", "day2_return", "day3_return"]
    result = {
        "base_id": base_hypothesis["id"], "confirmed_id": confirmed_hypothesis["id"],
        "n_confirmed_days": len(confirmed_dates), "n_unconfirmed_only_days": len(unconfirmed_only_dates),
    }
    for col in outcome_columns:
        result.setdefault("confirmed", {})[col] = compute_stats(
            _collect_day_level_returns(confirmed_dates, tickers, asx_outcomes_by_ticker, col, sign), costs)
        result.setdefault("unconfirmed_only", {})[col] = compute_stats(
            _collect_day_level_returns(unconfirmed_only_dates, tickers, asx_outcomes_by_ticker, col, sign), costs)
    return result


def leave_one_out_analysis(hypothesis, aligned_driver_table, asx_outcomes_by_ticker,
                            asx_theme_stocks, costs=None):
    """Does the basket-level result survive removing any ONE stock? For
    each ticker in the basket, recomputes the day-level (basket-average)
    result using only the OTHER tickers, per split. If dropping any
    single name collapses or flips the sign, the "basket" result was
    substantially that one name, not a broad effect — a stronger,
    more direct test than eyeballing per-stock consistency separately.
    """
    theme = hypothesis["theme"]
    direction = hypothesis["direction"]
    usable_from = pd.Timestamp(hypothesis["usable_from"])
    tickers = asx_theme_stocks[theme]
    sign = 1 if direction == "LONG" else -1

    driver_slice = aligned_driver_table[aligned_driver_table.index >= usable_from]
    matched_dates = [d for d in driver_slice.index if hypothesis["condition"](driver_slice.loc[d])]
    if not matched_dates:
        return {"hypothesis_id": hypothesis["id"], "n_matched_days": 0, "note": "No historical matches found"}

    train_d, val_d, test_d = chronological_split(matched_dates)
    splits = [("train", train_d), ("validation", val_d), ("test", test_d)]
    col = "open_to_close_pct"

    result = {"hypothesis_id": hypothesis["id"], "label": hypothesis["label"], "n_matched_days": len(matched_dates),
              "full_basket": {s: compute_stats(_collect_day_level_returns(d, tickers, asx_outcomes_by_ticker, col, sign), costs)
                              for s, d in splits}}

    for excluded in tickers:
        remaining = [t for t in tickers if t != excluded]
        result.setdefault("leave_one_out", {})[excluded] = {
            s: compute_stats(_collect_day_level_returns(d, remaining, asx_outcomes_by_ticker, col, sign), costs)
            for s, d in splits
        }
    return result


def regime_stability_analysis(hypothesis, aligned_driver_table, asx_outcomes_by_ticker,
                               asx_theme_stocks, costs=None, n_periods=4):
    """Divides the FULL matched history (not train/val/test — a
    different cut, by calendar time) into n_periods consecutive,
    roughly equal chunks, and checks whether the result holds up
    across each, or is concentrated in one narrow historical regime
    (e.g. a single commodity bull run). Diagnostic, not a new
    train/val/test — no threshold is chosen or touched here.
    """
    theme = hypothesis["theme"]
    direction = hypothesis["direction"]
    usable_from = pd.Timestamp(hypothesis["usable_from"])
    tickers = asx_theme_stocks[theme]
    sign = 1 if direction == "LONG" else -1

    driver_slice = aligned_driver_table[aligned_driver_table.index >= usable_from]
    matched_dates = sorted(d for d in driver_slice.index if hypothesis["condition"](driver_slice.loc[d]))
    if not matched_dates:
        return {"hypothesis_id": hypothesis["id"], "n_matched_days": 0, "note": "No historical matches found"}

    n = len(matched_dates)
    chunk_size = max(1, n // n_periods)
    periods = []
    for i in range(n_periods):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_periods - 1 else n
        chunk_dates = matched_dates[start:end]
        if not chunk_dates:
            continue
        periods.append({
            "period_index": i, "start_date": str(chunk_dates[0].date()), "end_date": str(chunk_dates[-1].date()),
            "n_days": len(chunk_dates),
            "stats": compute_stats(_collect_day_level_returns(chunk_dates, tickers, asx_outcomes_by_ticker, "open_to_close_pct", sign), costs),
        })

    signs = [1 if p["stats"].get("expectancy_net", 0) > 0 else (-1 if p["stats"].get("expectancy_net", 0) < 0 else 0)
             for p in periods if p["stats"].get("n", 0) > 0]
    all_same_sign = len(set(signs)) == 1 if signs else False

    return {"hypothesis_id": hypothesis["id"], "label": hypothesis["label"], "n_matched_days": n,
            "periods": periods, "consistent_across_all_periods": all_same_sign}


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
