"""
PHASE B CLASSIFICATION — pre-specified BEFORE any Phase B results exist
============================================================================
These constants and the classification rule below are fixed now, before
Phase B is run. They are not adjusted afterward based on what looks
good — that's the whole point of specifying them here, in their own
file, ahead of time.

Grading anchor: the +2% threshold at full-session is used as the single
representative cut for the A/B/C grade, chosen for being the moderate,
middle threshold (not the loosest=1% or the most extreme=5%) and the
full trading day (not an arbitrary early cutoff). This does NOT mean
other thresholds/checkpoints are hidden — every one of them (1/2/3/5%
x 10:15/10:30/11:00/12:00/full_session) is still reported in full in
the CSV output. The anchor exists only to produce ONE top-level grade
per (hypothesis, ticker) pair for ranking purposes.

Rule:
  C (unstable/no edge) — if the delta (signal probability minus
    baseline probability) has a DIFFERENT SIGN between validation and
    test, or is <=0 at both. A sign flip is definitionally instability,
    regardless of how good either individual split looks.
  A (strong/stable) — delta is POSITIVE at both validation AND test,
    AND >= GRADE_A_MIN_DELTA_PP at both, AND sample size >= GRADE_A_MIN_N
    at both splits.
  B (promising, needs more evidence) — delta is positive at both
    validation and test (same sign, real direction) but doesn't clear
    the A bar on magnitude or sample size.
"""

CLASSIFICATION_ANCHOR_THRESHOLD_PCT = 2.0
CLASSIFICATION_ANCHOR_CHECKPOINT = "full_session"
GRADE_A_MIN_DELTA_PP = 10.0
GRADE_A_MIN_N = 30


def classify_finding(val_delta_pp, test_delta_pp, n_val, n_test):
    """Returns 'A', 'B', or 'C'. All four inputs can be None (missing
    data) -> treated as C, not silently skipped."""
    if val_delta_pp is None or test_delta_pp is None or n_val is None or n_test is None:
        return "C"

    val_positive = val_delta_pp > 0
    test_positive = test_delta_pp > 0

    if val_positive != test_positive:
        return "C"  # sign flip between splits
    if not val_positive and not test_positive:
        return "C"  # negative or zero at both -- not a finding either way

    if (val_delta_pp >= GRADE_A_MIN_DELTA_PP and test_delta_pp >= GRADE_A_MIN_DELTA_PP
            and n_val >= GRADE_A_MIN_N and n_test >= GRADE_A_MIN_N):
        return "A"
    return "B"
