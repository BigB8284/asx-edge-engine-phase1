"""
V3 PILOT CONFIG — new hypothesis definitions for pilot tickers with no
existing coverage
================================================================
Of the 16 approved pilot tickers, most already have hypothesis
coverage from config_v2.py's existing baskets and need nothing new:
  CIA.AX  -> Iron Ore, H4/H4b (existing, frozen)
  ILU.AX, ARU.AX -> Rare Earths, H9 (existing)
  NXT.AX  -> ASX Technology, H10 (existing)
  GMG.AX  -> REITs, H12 (existing — INVERTED sign convention, see
             config_v2.py's driver_sign_convention field, carried
             through unchanged)
  BPT.AX, WDS.AX -> Energy, H1/H1b (existing, frozen)
  SFR.AX  -> Copper, H7 (existing)
  BOE.AX  -> Uranium, H6 (existing)
  CBA.AX  -> already IN config_v2's "Financials (thematic)" basket,
             tested under H11/H11b automatically, nothing new needed

Four pilot tickers genuinely need new definitions:
  QBE.AX, S32.AX -> new to any basket. Given their own theme names
             below (NOT added into the existing "Financials (thematic)"
             / Iron Ore baskets) so those established, already-tested
             baskets are never silently mutated — same driver logic
             reused, separate basket, separate audit trail.
  JHX.AX, NEM.AX, BHP.AX, RIO.AX -> the new ADR/dual-listing theme
             (section 10 of the spec). This is a genuinely different
             KIND of driver: instead of an independent commodity/index
             proxy, the driver IS the stock's own overseas-listed
             price. BHP.AX and RIO.AX ALSO keep their existing Iron
             Ore coverage via H4 unchanged — the ADR hypothesis below
             is a separate, additional test of the same ticker, not a
             replacement.

SCOPING NOTE: this first pilot version of the ADR hypothesis uses the
overseas overnight move directly (e.g. "NYSE:JHX moved >+2% overnight
-> ASX:JHX LONG signal"), matching the exact structural pattern already
used for every commodity driver. The fuller version described in the
spec — converting the overseas price to an AUD-equivalent implied ASX
price and testing whether the gap to the actual ASX open converges —
is a real, separate refinement worth doing once this simpler version
confirms the basic relationship is there at all. Building the full
convergence-gap analysis before knowing whether the underlying
relationship exists would be solving a harder problem before the easy
version has even been checked.
"""

from config_v2 import DRIVERS as DRIVERS_V2, ASX_THEME_STOCKS as ASX_THEME_STOCKS_V2, HYPOTHESES as HYPOTHESES_V2, _pair

# ---------------------------------------------------------------------------
# NEW DRIVERS — the four ADR names' own overseas listings
# ---------------------------------------------------------------------------
# Format matches config_v2.py's convention exactly: (yfinance_ticker, role, start_date, note)
PILOT_DRIVERS_ADDITIONS = {
    "jhx_nyse": ("JHX", "PRIMARY", "2025-01-01",
                 "James Hardie's NYSE line became its PRIMARY listing during 2025 post-AZEK acquisition; "
                 "ASX:JHX is now a secondary CDI. usable_from set to reflect this, not the project's usual 2020-10-12 start."),
    "nem_nyse": ("NEM", "PRIMARY", "2023-11-01",
                 "Newmont's NYSE line, driver for the ASX:NEM CDI. usable_from set to Nov 2023: config_v2.py's own "
                 "comment records ASX:NEM as 'near-empty' before this (CDI trading only became meaningful post-Newcrest merger)."),
    "bhp_adr": ("BHP", "PRIMARY", "2020-10-12", None),  # NYSE:BHP ADR, same ticker symbol as ASX
    "rio_adr": ("RIO", "PRIMARY", "2020-10-12", None),  # NYSE:RIO ADR, same ticker symbol as ASX
}
DRIVERS = {**DRIVERS_V2, **PILOT_DRIVERS_ADDITIONS}

# ---------------------------------------------------------------------------
# NEW BASKETS — kept separate from existing established themes
# ---------------------------------------------------------------------------
PILOT_THEME_STOCKS_ADDITIONS = {
    "Financials (thematic) - pilot addition": ["QBE.AX"],
    "Iron Ore - pilot addition": ["S32.AX"],
    # Each ADR ticker gets its OWN theme/basket, not a shared one — its
    # driver is that specific stock's own overseas price, so a shared
    # basket would (and, caught during testing, DID) incorrectly match
    # every ADR ticker to every ADR hypothesis.
    "ADR / Dual-listing - JHX": ["JHX.AX"],
    "ADR / Dual-listing - NEM": ["NEM.AX"],
    "ADR / Dual-listing - BHP": ["BHP.AX"],
    "ADR / Dual-listing - RIO": ["RIO.AX"],
}
ASX_THEME_STOCKS = {**ASX_THEME_STOCKS_V2, **PILOT_THEME_STOCKS_ADDITIONS}

# ---------------------------------------------------------------------------
# NEW HYPOTHESES
# ---------------------------------------------------------------------------
PILOT_HYPOTHESES_ADDITIONS = []

# QBE: same driver logic as H11 (US financials + broad market confirming),
# separate basket so the existing Financials theme is untouched.
PILOT_HYPOTHESES_ADDITIONS += _pair(
    "H11_pilot_qbe", "US Financials (XLF) +-1% -> QBE (pilot addition)",
    "Financials (thematic) - pilot addition", ["xlf"], "1998-12-22",
    cond_long=lambda row: row.get("xlf", float("nan")) > 1.0,
    cond_short=lambda row: row.get("xlf", float("nan")) < -1.0,
    status="experimental",  # weaker/more diffuse driver relationship than a commodity name, flagged per prior assessment
)

# S32: same driver logic as H4 (iron ore), separate basket. S32 is a
# multi-commodity basket (iron ore/manganese/alumina) so this tests
# only the iron-ore-exposure slice of its driver relationship, not the
# whole story — flagged as experimental for that reason too.
PILOT_HYPOTHESES_ADDITIONS += _pair(
    "H4_pilot_s32", "Iron ore +-2% -> S32 (pilot addition, iron-ore-exposure slice only)",
    "Iron Ore - pilot addition", ["iron_ore"], "2020-10-12",
    cond_long=lambda row: row.get("iron_ore", float("nan")) > 2.0,
    cond_short=lambda row: row.get("iron_ore", float("nan")) < -2.0,
    status="experimental",
)

# ADR / Dual-listing — four separate hypotheses, one per ticker, since
# each one's driver is that SPECIFIC stock's own overseas price, not a
# shared commodity/index. usable_from varies per ticker (see DRIVERS
# above) to reflect real listing-history differences, not copy-pasted.
PILOT_HYPOTHESES_ADDITIONS += _pair(
    "H_adr_jhx", "NYSE:JHX overnight +-2% -> ASX:JHX", "ADR / Dual-listing - JHX", ["jhx_nyse"], "2025-01-01",
    cond_long=lambda row: row.get("jhx_nyse", float("nan")) > 2.0,
    cond_short=lambda row: row.get("jhx_nyse", float("nan")) < -2.0,
)
PILOT_HYPOTHESES_ADDITIONS += _pair(
    "H_adr_nem", "NYSE:NEM overnight +-2% -> ASX:NEM", "ADR / Dual-listing - NEM", ["nem_nyse"], "2023-11-01",
    cond_long=lambda row: row.get("nem_nyse", float("nan")) > 2.0,
    cond_short=lambda row: row.get("nem_nyse", float("nan")) < -2.0,
)
PILOT_HYPOTHESES_ADDITIONS += _pair(
    "H_adr_bhp", "NYSE:BHP ADR overnight +-2% -> ASX:BHP (in addition to existing H4 iron-ore test)",
    "ADR / Dual-listing - BHP", ["bhp_adr"], "2020-10-12",
    cond_long=lambda row: row.get("bhp_adr", float("nan")) > 2.0,
    cond_short=lambda row: row.get("bhp_adr", float("nan")) < -2.0,
)
PILOT_HYPOTHESES_ADDITIONS += _pair(
    "H_adr_rio", "NYSE:RIO ADR overnight +-2% -> ASX:RIO (in addition to existing H4 iron-ore test)",
    "ADR / Dual-listing - RIO", ["rio_adr"], "2020-10-12",
    cond_long=lambda row: row.get("rio_adr", float("nan")) > 2.0,
    cond_short=lambda row: row.get("rio_adr", float("nan")) < -2.0,
)

HYPOTHESES = HYPOTHESES_V2 + PILOT_HYPOTHESES_ADDITIONS

# ---------------------------------------------------------------------------
# THE 16 APPROVED PILOT TICKERS
# ---------------------------------------------------------------------------
PILOT_TICKERS = [
    "CIA.AX", "ILU.AX", "ARU.AX", "NXT.AX", "GMG.AX",   # previous Grade A winners
    "BPT.AX", "SFR.AX", "WDS.AX", "BOE.AX",              # previous old-anchor false rejects
    "CBA.AX", "QBE.AX", "S32.AX",                        # newly added, untested
    "JHX.AX", "NEM.AX", "BHP.AX", "RIO.AX",              # ADR/dual-listing candidates
]


def hypotheses_for_ticker(ticker):
    """Every hypothesis whose basket includes this ticker, across every
    theme it belongs to. A ticker can legitimately appear under more
    than one hypothesis (e.g. BHP.AX under both H4 and H_adr_bhp) —
    both get tested and reported, not silently merged into one."""
    relevant = []
    for h in HYPOTHESES:
        if ticker in ASX_THEME_STOCKS.get(h["theme"], []):
            relevant.append(h)
    return relevant
