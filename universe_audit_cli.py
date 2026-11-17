"""
Headless runner for the universe audit - same code as the Streamlit app.
Used by .github/workflows/universe_audit.yml so the long first 5m download
can run on GitHub's servers (up to 6 hours) instead of a phone browser tab.

Env vars: EODHD_API_TOKEN, DATA_REPO, DATA_REPO_TOKEN
Usage:    python universe_audit_cli.py --budget-min 300
"""
import argparse
import json
import os
import sys

import universe_audit_core as core
from data_store import make_store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-min", type=float, default=300)
    ap.add_argument("--reserve-calls", type=int, default=2000)
    args = ap.parse_args()

    token = os.environ.get("EODHD_API_TOKEN")
    if not token:
        sys.exit("EODHD_API_TOKEN is not set.")
    store, warn = make_store(os.environ.get("DATA_REPO"), os.environ.get("DATA_REPO_TOKEN"))
    if warn:
        print("WARNING:", warn)
    print("Storage:", store.describe())

    out = core.run_audit(core.EODHDClient(token), store, time_budget_s=args.budget_min * 60,
                         reserve_calls=args.reserve_calls,
                         progress=lambda f, m: None, log=lambda m: print(m, flush=True))
    if not out.get("ok"):
        sys.exit(out.get("stop_reason"))
    os.makedirs("out", exist_ok=True)
    with open(os.path.join("out", out["csv_name"]), "wb") as f:
        f.write(out["csv_bytes"])
    print(json.dumps({"complete": out["complete"], "stop_reason": out["stop_reason"],
                      "summary": out["summary"], "calls": out["meta"]["total_calls_this_run"]}, indent=2))


if __name__ == "__main__":
    main()
