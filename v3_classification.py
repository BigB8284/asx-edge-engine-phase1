"""
V3 CLASSIFICATION — sweet-spot scoring + confidence labels
================================================================
This is the NEW system for V3, separate from and not replacing
phase_b_classification.py. That file's A/B/C grade at the single
2%/full_session anchor is preserved untouched for audit/history, per
approved instruction — it has zero influence on eligibility here.

Everything in this file is FROZEN per explicit approval (2026-08-14):
  - The scoring formula and its exact constants (0.3 exponent, the five
    speed-bonus values) are human preference settings reflecting how
    Brent actually wants to trade, not parameters to optimise.
  - They must NOT be changed later because a different value would have
    produced better-looking historical results. If they ever change,
    that has to be a fresh, explicit decision — not a silent tune.

Two clearly separate things live in this file, and they answer two
different questions:
  1. OPPORTUNITY SCORE (score_opportunity) — among a stock's ELIGIBLE
     setups, which one is the recommended ("sweet spot")? This is a
     preference-ordering tool. It is NOT a probability and must never
     be displayed in a way that could be confused for one.
  2. CONFIDENCE LABEL (classify_confidence) — regardless of which setup
     is the sweet spot, how much should that pick actually be trusted?
     Driven entirely by sample size and validation/test stability, not
     by the score.
"""

import math

# ---------------------------------------------------------------------------
# ELIGIBILITY GATE — applied on TRAIN data only, before scoring
# ---------------------------------------------------------------------------
GATE_MIN_N = 30
GATE_MIN_DELTA_PP = 15.0  # raised from the old system's 10pp: the search
                          # space here is ~6x wider (4 thresholds x 5
                          # scanner checkpoints vs the old single anchor),
                          # so the bar for "real" needs to be higher too.


def passes_eligibility_gate(train_delta_pp, train_n, mae_before_reached, threshold_pct):
    """A (threshold, checkpoint) combo is only a candidate for scoring if,
    on TRAIN data: n >= 30, delta >= +15pp, and the median adverse
    excursion before the target was typically reached is smaller than
    the target itself (a setup you'd typically get stopped out of
    before it pays off shouldn't be a candidate at all).

    mae_before_reached: signed, e.g. -0.8 means "typically ran 0.8%
    against the signal before the target was reached". None (never
    reached on train) fails the gate.
    """
    if train_delta_pp is None or train_n is None or mae_before_reached is None:
        return False
    if train_n < GATE_MIN_N:
        return False
    if train_delta_pp < GATE_MIN_DELTA_PP:
        return False
    if abs(mae_before_reached) >= threshold_pct:
        return False
    return True


# ---------------------------------------------------------------------------
# OPPORTUNITY SCORE — FROZEN 2026-08-14, do not tune against results
# ---------------------------------------------------------------------------
MAGNITUDE_EXPONENT = 0.3  # <1 = diminishing returns on bigger targets,
                          # so probability dominates rather than magnitude
SPEED_BONUS = {
    "10:15": 1.05,
    "10:30": 1.03,
    "11:00": 1.01,
    "11:30": 1.00,
    "12:00": 1.00,
    # full_session deliberately absent — it's research-only, never a
    # scanner-eligible checkpoint, so it can never be scored or selected
    # as a sweet spot. See NOT_SCANNER_ELIGIBLE below.
}
NOT_SCANNER_ELIGIBLE_CHECKPOINTS = {"full_session"}


def score_opportunity(probability_pct, threshold_pct, checkpoint):
    """probability_pct: 0-100 (the TRAIN signal probability for this
    combo). Returns None if the checkpoint isn't scanner-eligible
    (i.e. full_session) — such combos can be displayed in a stock's
    full ladder for reference but can never carry a score or be
    selected as the sweet spot.

    NOT a probability. Never display this next to a probability number
    without a label distinguishing the two.
    """
    if checkpoint not in SPEED_BONUS:
        return None
    return probability_pct * (threshold_pct ** MAGNITUDE_EXPONENT) * SPEED_BONUS[checkpoint]


def select_sweet_spot(candidates):
    """candidates: list of dicts, each with at minimum
    {"threshold_pct", "checkpoint", "train_probability_pct", "score"}
    for combos that already passed passes_eligibility_gate(). Returns
    the single highest-scoring candidate, or None if the list is empty.
    Ties broken by earliest checkpoint (SPEED_BONUS ordering), then by
    threshold_pct ascending (prefer the smaller/more probable of two
    genuinely tied setups) — deterministic, no randomness.
    """
    if not candidates:
        return None
    checkpoint_order = list(SPEED_BONUS.keys())
    return max(
        candidates,
        key=lambda c: (
            c["score"],
            -checkpoint_order.index(c["checkpoint"]) if c["checkpoint"] in checkpoint_order else -99,
            -c["threshold_pct"],
        ),
    )


# ---------------------------------------------------------------------------
# CONFIDENCE LABEL — separate axis from the score entirely
# ---------------------------------------------------------------------------
CONFIDENCE_MIN_N = 30
CONFIDENCE_MIN_DELTA_PP = 15.0
LOW_SAMPLE_N = 20  # below this on val or test: insufficient evidence,
                   # not "no edge" — see EXPERIMENTAL below


def classify_confidence(val_delta_pp, test_delta_pp, n_val, n_test):
    """Returns one of VALIDATED / WATCH / EXPERIMENTAL / NO EDGE.
    Mirrors the clean structure of the old classify_finding() in
    phase_b_classification.py but with V3's own (higher) bar and
    four-tier structure — this does NOT call or modify that function;
    the two systems are intentionally independent.

    All four inputs may be None (missing data on a split) -> treated
    as EXPERIMENTAL (insufficient evidence), not silently dropped and
    not NO EDGE (which requires having actually looked and found
    nothing, not having failed to look at all).
    """
    if val_delta_pp is None or test_delta_pp is None or n_val is None or n_test is None:
        return "EXPERIMENTAL"

    if n_val < LOW_SAMPLE_N or n_test < LOW_SAMPLE_N:
        # Too little evidence to judge either way, regardless of how the
        # numbers look — "insufficient sample", never "no edge".
        return "EXPERIMENTAL"

    val_positive = val_delta_pp > 0
    test_positive = test_delta_pp > 0

    if val_positive != test_positive:
        # Sign flip. With adequate sample on both sides this is a real
        # negative finding, not a data problem.
        return "NO EDGE"
    if not val_positive and not test_positive:
        return "NO EDGE"

    # Same sign, both positive. Between full CONFIDENCE_MIN_N/DELTA and
    # LOW_SAMPLE_N sits WATCH — real direction, not yet enough to trust.
    if (val_delta_pp >= CONFIDENCE_MIN_DELTA_PP and test_delta_pp >= CONFIDENCE_MIN_DELTA_PP
            and n_val >= CONFIDENCE_MIN_N and n_test >= CONFIDENCE_MIN_N):
        return "VALIDATED"
    return "WATCH"
