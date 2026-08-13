"""
CONFIG V2 — BROAD SCANNER EXPANSION
=====================================
Extends config_v1.py, never modifies it. Iron Ore and Energy hypotheses
(H1/H1b/H4/H4b) are imported from V1 completely unchanged — frozen,
per instruction. Everything below is new: Gold gets a corrected basket
(V1's GOR.AX failed entirely, NEM.AX was nearly empty — this is a data
completeness fix, not a performance-chasing change, so it gets its own
clearly-named hypotheses rather than silently overwriting H2/H2b's
historical record). Eight new themes are added. Lithium gets one new
confirmed-pair hypothesis (H3c) alongside the existing frozen H3/H3b.

Two new fields appear on hypotheses that need them:
  - "driver_sign_convention": present ONLY where a driver's economic
    relationship is inverted (currently just REITs — falling yield is
    the bullish signal). Printed on every output table for that
    hypothesis so an inverted sign can't silently read as a mistake
    or, worse, get flipped back "to look normal."
  - "status": "experimental" on hypotheses where the driver-to-basket
    link is acknowledged to be weaker than the commodity-future-based
    themes (currently just Coal, where BTU is a loose thermal/met-coal
    proxy for the ASX names). Carried through to every report output.
    Hypotheses without this field are implicitly "standard".
"""

from config_v1 import (
    DRIVERS as DRIVERS_V1, ASX_THEME_STOCKS as ASX_THEME_STOCKS_V1,
    HYPOTHESES as HYPOTHESES_V1, COSTS,
)

# ---------------------------------------------------------------------------
# NEW DRIVERS (validated 13 Aug 2026 — ticker_validation_results_v2.csv)
# ---------------------------------------------------------------------------
DRIVERS_V2_ADDITIONS = {
    "rare_earths": ("REMX", "PRIMARY", "2010-10-28", None),
    "us10y_yield": ("^TNX", "PRIMARY", "1962-01-02",
                     "Yahoo quotes this as yield x10 (45.0 = 4.50%) — verify against a known "
                     "current yield when the driver table is first pulled, before trusting any result."),
}
DRIVERS = {**DRIVERS_V1, **DRIVERS_V2_ADDITIONS}

# ---------------------------------------------------------------------------
# NEW / CORRECTED ASX BASKETS (validated 13 Aug 2026)
# ---------------------------------------------------------------------------
ASX_THEME_STOCKS_V2_ADDITIONS = {
    "Gold (v2 corrected basket)": ["NST.AX", "EVN.AX", "RRL.AX", "RMS.AX"],  # GOR.AX (failed) and NEM.AX (near-empty) dropped
    "Uranium": ["PDN.AX", "BOE.AX", "DYL.AX"],  # BOE/DYL not yet individually spot-checked, see note above
    "Copper": ["SFR.AX", "29M.AX"],  # thin, 2-stock basket — reported per-stock, never pooled-only
    "Coal": ["WHC.AX", "YAL.AX", "NHC.AX"],
    "Rare Earths": ["LYC.AX", "ILU.AX", "ARU.AX"],  # ILU has mixed mineral-sands/REE exposure, not a pure play
    "ASX Technology": ["XRO.AX", "WTC.AX", "TNE.AX", "NXT.AX"],
    "Financials (thematic)": ["CBA.AX", "WBC.AX", "NAB.AX", "ANZ.AX", "MQG.AX"],  # distinct from H5's role as baseline/control
    "REITs": ["GMG.AX", "SGP.AX", "SCG.AX"],
}
ASX_THEME_STOCKS = {**ASX_THEME_STOCKS_V1, **ASX_THEME_STOCKS_V2_ADDITIONS}


def _pair(id_base, label_base, theme, drivers_used, usable_from, cond_long, cond_short,
          driver_sign_convention=None, status=None):
    base = {"theme": theme, "drivers_used": drivers_used, "usable_from": usable_from}
    if driver_sign_convention:
        base["driver_sign_convention"] = driver_sign_convention
    if status:
        base["status"] = status
    return [
        {**base, "id": f"{id_base}_long", "label": f"{label_base} (LONG)", "direction": "LONG", "condition": cond_long},
        {**base, "id": f"{id_base}_short", "label": f"{label_base} (SHORT)", "direction": "SHORT", "condition": cond_short},
    ]


HYPOTHESES_V2_ADDITIONS = []

# --- Gold, corrected basket (v1's H2/H2b stay untouched in config_v1.py) ---
HYPOTHESES_V2_ADDITIONS += _pair(
    "H2v2", "Gold +-2% -> ASX Gold basket (corrected)", "Gold (v2 corrected basket)", ["gold"], "2006-05-22",
    cond_long=lambda row: row.get("gold", float("nan")) > 2.0,
    cond_short=lambda row: row.get("gold", float("nan")) < -2.0,
)
HYPOTHESES_V2_ADDITIONS += _pair(
    "H2v2b", "Gold +-2% AND US Gold Miners (GDX) confirming -> ASX Gold basket (corrected)",
    "Gold (v2 corrected basket)", ["gold", "gdx"], "2006-05-22",
    cond_long=lambda row: row.get("gold", float("nan")) > 2.0 and row.get("gdx", float("nan")) > 2.0,
    cond_short=lambda row: row.get("gold", float("nan")) < -2.0 and row.get("gdx", float("nan")) < -2.0,
)

# --- Uranium ---
HYPOTHESES_V2_ADDITIONS += _pair(
    "H6", "Uranium (URA) +-2% -> ASX Uranium basket", "Uranium", ["ura"], "2010-11-05",
    cond_long=lambda row: row.get("ura", float("nan")) > 2.0,
    cond_short=lambda row: row.get("ura", float("nan")) < -2.0,
)

# --- Copper ---
HYPOTHESES_V2_ADDITIONS += _pair(
    "H7", "Copper +-2% -> ASX Copper basket", "Copper", ["copper"], "2004-03-03",  # capped to SFR.AX's start, the shorter of the 2-stock basket
    cond_long=lambda row: row.get("copper", float("nan")) > 2.0,
    cond_short=lambda row: row.get("copper", float("nan")) < -2.0,
)

# --- Coal (EXPERIMENTAL: BTU is a loose thermal-coal-stock proxy, not a direct ASX-relevant benchmark) ---
HYPOTHESES_V2_ADDITIONS += _pair(
    "H8", "US Coal proxy (BTU) +-3% -> ASX Coal basket", "Coal", ["coal"], "2017-04-03",
    cond_long=lambda row: row.get("coal", float("nan")) > 3.0,
    cond_short=lambda row: row.get("coal", float("nan")) < -3.0,
    status="experimental",
)

# --- Rare Earths ---
HYPOTHESES_V2_ADDITIONS += _pair(
    "H9", "Rare Earths (REMX) +-3% -> ASX Rare Earths basket", "Rare Earths", ["rare_earths"], "2010-10-28",
    cond_long=lambda row: row.get("rare_earths", float("nan")) > 3.0,
    cond_short=lambda row: row.get("rare_earths", float("nan")) < -3.0,
)

# --- ASX Technology ---
HYPOTHESES_V2_ADDITIONS += _pair(
    "H10", "Nasdaq +-1.5% -> ASX Technology basket", "ASX Technology", ["nasdaq"], "1971-02-05",
    cond_long=lambda row: row.get("nasdaq", float("nan")) > 1.5,
    cond_short=lambda row: row.get("nasdaq", float("nan")) < -1.5,
)

# --- Financials, thematic (distinct from H5's role as baseline/control) ---
HYPOTHESES_V2_ADDITIONS += _pair(
    "H11", "US Financials (XLF) +-1% -> ASX Financials basket", "Financials (thematic)", ["xlf"], "1998-12-22",
    cond_long=lambda row: row.get("xlf", float("nan")) > 1.0,
    cond_short=lambda row: row.get("xlf", float("nan")) < -1.0,
)
HYPOTHESES_V2_ADDITIONS += _pair(
    "H11b", "US Financials (XLF) +-1% AND broad S&P500 confirming -> ASX Financials basket",
    "Financials (thematic)", ["xlf", "sp500"], "1998-12-22",
    cond_long=lambda row: row.get("xlf", float("nan")) > 1.0 and row.get("sp500", float("nan")) > 0.5,
    cond_short=lambda row: row.get("xlf", float("nan")) < -1.0 and row.get("sp500", float("nan")) < -0.5,
)

# --- REITs (INVERTED sign convention: falling yield is the bullish signal) ---
HYPOTHESES_V2_ADDITIONS += _pair(
    "H12", "US 10yr yield falling/rising -> ASX REITs basket", "REITs", ["us10y_yield"], "1988-04-29",  # capped to SGP.AX's start
    cond_long=lambda row: row.get("us10y_yield", float("nan")) < -2.5,   # yield FALLING -> REIT LONG
    cond_short=lambda row: row.get("us10y_yield", float("nan")) > 2.5,  # yield RISING -> REIT SHORT
    driver_sign_convention="INVERTED — LONG triggers on yield FALLING (< -2.5%), SHORT triggers on yield RISING (> +2.5%). "
                            "This is the opposite sign convention to every other hypothesis in this project — check twice before editing.",
)

# --- Lithium: new confirmed-pair hypothesis alongside the frozen H3/H3b ---
HYPOTHESES_V2_ADDITIONS += _pair(
    "H3c", "Lithium (LIT) +-3% AND Albemarle/SQM confirming -> ASX Lithium basket",
    "Lithium", ["lit", "albemarle", "sqm"], "2010-07-23",
    cond_long=lambda row: row.get("lit", float("nan")) > 3.0 and (
        (row.get("albemarle", float("nan")) > 2.0) or (row.get("sqm", float("nan")) > 2.0)),
    cond_short=lambda row: row.get("lit", float("nan")) < -3.0 and (
        (row.get("albemarle", float("nan")) < -2.0) or (row.get("sqm", float("nan")) < -2.0)),
)

# ---------------------------------------------------------------------------
# COMBINED HYPOTHESIS SET: V1's frozen Iron Ore/Energy + V1's frozen
# Lithium (H3/H3b) + everything new above. V1's Gold/Financials-control
# (H2/H2b/H5) are intentionally NOT carried into this combined list —
# H2/H2b are superseded by the corrected-basket H2v2/H2v2b, and H5 stays
# in config_v1.py in its own role as a standalone baseline/control, not
# part of the opportunity-ranking scanner.
# ---------------------------------------------------------------------------
_FROZEN_V1_IDS_TO_KEEP = {
    "H1_long", "H1_short", "H1b_long", "H1b_short",     # Energy — frozen
    "H4_long", "H4_short", "H4b_long", "H4b_short",     # Iron Ore — frozen
    "H3_long", "H3_short", "H3b_long", "H3b_short",     # Lithium — frozen
}
HYPOTHESES = [h for h in HYPOTHESES_V1 if h["id"] in _FROZEN_V1_IDS_TO_KEEP] + HYPOTHESES_V2_ADDITIONS
