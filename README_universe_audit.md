# ASX Universe Audit: setup and running

## Files (add to `asx-edge-engine-phase1`, nothing existing is changed)
| File | What it is |
|---|---|
| `universe_audit_app.py` | The Streamlit app (main file) |
| `universe_audit_core.py` | Audit logic: symbols → daily screen → 5m check → results |
| `data_store.py` | Permanent cache in a private GitHub repo (Release assets) |
| `universe_audit_cli.py` | Same audit, headless (used by GitHub Actions) |
| `.github/workflows/universe_audit.yml` | Optional: run the long first download on GitHub's servers |
| `tests/` | Offline tests (`python -m pytest -q tests`), optional to upload |

`requirements.txt` already has everything (`pyarrow` is in there).

## One-time setup (about 10 minutes)
1. **Private data repo:** github.com/new → name `asx-data-store` → **Private** → tick **Add a README file** → Create.
2. **Token for it:** GitHub → Settings → Developer settings → Personal access tokens → **Fine-grained tokens** → Generate.
   Repository access: *Only select repositories* → `asx-data-store`. Permissions → Repository → **Contents: Read and write**. Copy the token.
3. **Streamlit app:** share.streamlit.io → Create app → repo `BigB8284/asx-edge-engine-phase1`, branch `main`,
   main file `universe_audit_app.py` → Advanced settings → Secrets:
   ```toml
   EODHD_API_TOKEN = "your-eodhd-token"
   DATA_REPO = "BigB8284/asx-data-store"
   DATA_REPO_TOKEN = "github_pat_..."
   ```
4. Open the app. You should see a green **"Cache: private GitHub repo … survives redeploys"** banner.

## Running
- Press **▶ Run / resume audit**. Each press runs for up to 20 minutes (adjustable in Settings), then saves and stops.
- Keep pressing until it says **Audit complete**. Nothing is downloaded twice: stocks already checked cost 0 calls.
- Download the CSV with the big button. Every stock is in it with its status and reasons at both $1M and $500k.
- Page reload or redeploy: the last results load straight from the data repo.

**Optional faster first run:** in `asx-edge-engine-phase1` → Settings → Secrets and variables → Actions, add the same
three secrets (`EODHD_API_TOKEN`, `DATA_REPO`, `DATA_REPO_TOKEN`). Then Actions tab → *ASX universe audit (headless)* →
Run workflow. It runs up to ~5 hours in one go, into the same cache. Don't run it and the app at the same time.

## Statuses
| Status | Meaning |
|---|---|
| `ELIGIBLE` | Passes everything incl. ≥400 usable 5m sessions → initial model training + live scanning |
| `LIVE_ONLY_BUILDING_HISTORY` | Liquid and clean but newer → scanned live, joins training at 400 sessions |
| `WATCH_NEW_LISTING` | Under 20 sessions since listing → re-checked every audit, never permanently excluded |
| `PENDING_5M_CHECK` | Passed daily screen, 5m not checked yet (or a fetch failed) → press Run again |
| `EXCLUDED` | Fails a rule; the reason is in `reasons_1m` / `reasons_500k` |

## Rules (all adjustable in Settings)
Ordinary shares only: EODHD type *Common Stock*, AUD, 3-character ASX code, and a name that doesn't look like a
right, option, note, ETF and so on. Last close ≥ $1. Median daily turnover (close × volume, untraded days = $0) over
the last 60 sessions ≥ $1M, compared against $500k. Traded on ≥90% of sessions. Not suspended (traded within the last
5 sessions). 5m data: ≥400 usable sessions, using the V3 engine's own cleaning, and ≥80% of the 5-minute
slots between 10:00 and 11:00 traded over the last 60 sessions. All of those sessions are after ASX's single open
(23 June 2025), so the full first hour is fair for every code.

## API cost (EODHD documented costs: bulk 100, intraday 5, symbols 1, /user 0)
- First daily screen: ~61 bulk files ≈ 6,100 calls (one-off; later audits only fetch new days).
- 5m history from Oct 2020: 4 requests = 20 calls per stock, only for stocks passing the $500k daily screen.
- Exact numbers for your universe are shown in the app's **Costs & storage** tab after the first run.
