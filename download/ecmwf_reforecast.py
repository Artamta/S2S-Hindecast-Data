#!/usr/bin/env python3
"""
ecmwf_reforecast.py
===================
Download ECMWF S2S reforecasts (hindcasts) for climatology computation.

Target variables (6 — paper-relevant):
  Surface  : tp, 2t, mx2t, mn2t, msl
  Pressure : z @ 500 hPa (Z500)

Forecast types : control (cf) + perturbed (pf) members
Lead time      : day 1 – 90  (steps 24, 48, … 2160 h)
Hindcast period: 2000 – 2019  (20 years, packed per request)

Init date schedule (CY49R1+, operational since 12 Nov 2024):
  Days 1 / 5 / 9 / 13 / 17 / 21 / 25 / 29 of each month
  → 8 dates/month × 12 months × 20 years = 1900 init dates total

Output layout
-------------
  /storage/raj.ayush/All_Model_Data/ecmwf/reforecasts/
  ├── tp/        <YYYYMMDD>_cf.nc   <YYYYMMDD>_pf.nc  ...
  ├── 2t/
  ├── mx2t/
  ├── mn2t/
  ├── msl/
  └── z/
      └── 500/   <YYYYMMDD>_cf.nc  ...

Logs
----
  /storage/raj.ayush/s2s-data-pipeline/logs/ecmwf/
  ├── download_YYYYMMDD_HHMMSS.log   ← human-readable run log
  └── requests.jsonl                 ← one JSON record per completed file

Usage
-----
  # Dry-run on login node (always do this first)
  python download/ecmwf_reforecast.py --dry-run

  # Single date test (verify API works)
  python download/ecmwf_reforecast.py --date 2000-01-01 --dry-run
  python download/ecmwf_reforecast.py --date 2000-01-01

  # Full download (submit via SLURM — see slurm/ecmwf_download.sbatch)
  python download/ecmwf_reforecast.py --workers 4

  # Surface only
  python download/ecmwf_reforecast.py --sfc-only

  # Control forecast only (no perturbed members)
  python download/ecmwf_reforecast.py --no-pf

Requirements
------------
  conda activate s2s-hind
  ~/.cdsapirc  →  url: https://ecds.ecmwf.int/api
                  key: <your-key>
  cdsapi >= 0.7.7
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cdsapi
import pandas as pd

# ── PATHS ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/storage/raj.ayush/All_Model_Data/ecmwf/reforecasts")
LOG_DIR  = Path("/storage/raj.ayush/s2s-data-pipeline/logs/ecmwf")

# ── HINDCAST PERIOD ───────────────────────────────────────────────────────────
YEAR_START = 2000
YEAR_END   = 2019

# CY49R1+ reforecast init schedule: fixed days of month
REFORECAST_DAYS = {1, 5, 9, 13, 17, 21, 25, 29}

# ── FORECAST LEAD TIME ────────────────────────────────────────────────────────
# 90 days at 24-hour intervals
STEPS = [str(h) for h in range(24, 90 * 24 + 1, 24)]   # "24","48",...,"2160"

# ── VARIABLES ─────────────────────────────────────────────────────────────────
# 5 surface + Z500 = 6 paper-relevant variables
SURFACE_VARS = {
    "total_precipitation":                          "tp",
    "2m_temperature":                               "2t",
    "maximum_2m_temperature_in_the_last_24_hours":  "mx2t",
    "minimum_2m_temperature_in_the_last_24_hours":  "mn2t",
    "mean_sea_level_pressure":                      "msl",
}

# Only Z500 at pressure level (most relevant for S2S dynamics)
PL_VARS = {
    "geopotential": "z",
}
PRESSURE_LEVELS = ["500"]

# ── FORECAST TYPES ────────────────────────────────────────────────────────────
FTYPES = {
    "cf": "control_reforecast",
    "pf": "perturbed_reforecast",
}

# ── RETRY ─────────────────────────────────────────────────────────────────────
MAX_RETRIES   = 5
RETRY_BACKOFF = [30, 60, 120, 300, 600]   # seconds


# ── HELPERS ───────────────────────────────────────────────────────────────────
def setup_logging(run_id: str) -> tuple[logging.Logger, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"download_{run_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger(__name__)
    return log, log_file


def outpath(short_name: str, date: pd.Timestamp, ftype_key: str,
            level: str | None = None) -> Path:
    d = DATA_DIR / short_name / level if level else DATA_DIR / short_name
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{date:%Y%m%d}_{ftype_key}.nc"


def is_done(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def generate_init_dates(year_start: int, year_end: int) -> list[pd.Timestamp]:
    all_days = pd.date_range(f"{year_start}-01-01", f"{year_end}-12-31", freq="D")
    return [
        d for d in all_days
        if d.day in REFORECAST_DAYS and not (d.month == 2 and d.day == 29)
    ]


def build_tasks(dates: list[pd.Timestamp], ftypes: list[str],
                do_sfc: bool, do_pl: bool) -> list[dict]:
    tasks = []
    for date in dates:
        for ftype_key in ftypes:
            base = {
                "origin":        "ecmwf",
                "forecast_type": FTYPES[ftype_key],
                "year":          str(date.year),
                "month":         f"{date.month:02d}",
                "day":           f"{date.day:02d}",
                "time":          "00:00",
                "step":          STEPS,
                "data_format":   "netcdf",
            }
            if do_sfc:
                for cds_name, short in SURFACE_VARS.items():
                    p = outpath(short, date, ftype_key)
                    tasks.append({
                        "out":   str(p),
                        "req":   {**base, "level_type": "single_level", "variable": cds_name},
                        "label": f"{short}/{date:%Y%m%d}_{ftype_key}.nc",
                    })
            if do_pl:
                for cds_name, short in PL_VARS.items():
                    for lev in PRESSURE_LEVELS:
                        p = outpath(short, date, ftype_key, lev)
                        tasks.append({
                            "out":   str(p),
                            "req":   {**base, "level_type": "pressure_level",
                                      "variable": cds_name, "level": lev},
                            "label": f"{short}/{lev}/{date:%Y%m%d}_{ftype_key}.nc",
                        })
    return tasks


# ── DOWNLOAD WORKER ───────────────────────────────────────────────────────────
def download_one(task: dict, log: logging.Logger, req_log: Path) -> dict:
    out = Path(task["out"])

    if is_done(out):
        log.info(f"SKIP   {task['label']}")
        return {**task, "status": "skipped"}

    client   = cdsapi.Client(quiet=True)
    last_exc = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(f"START  {task['label']}  (attempt {attempt}/{MAX_RETRIES})")
            t0 = time.time()
            client.retrieve("s2s-reforecasts", task["req"], str(out))
            elapsed = time.time() - t0
            size_mb = out.stat().st_size / 1024 ** 2
            log.info(f"DONE   {task['label']}  {size_mb:.1f} MB  {elapsed:.0f}s")
            rec = {**task, "status": "success", "size_mb": round(size_mb, 2),
                   "elapsed_s": round(elapsed, 1), "ts": datetime.utcnow().isoformat()}
            with open(req_log, "a") as f:
                f.write(json.dumps(rec) + "\n")
            return rec
        except Exception as exc:
            last_exc = exc
            log.warning(f"RETRY  {task['label']}  attempt {attempt}: {exc}")
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF[attempt - 1]
                log.info(f"       waiting {wait}s ...")
                time.sleep(wait)

    log.error(f"FAIL   {task['label']} after {MAX_RETRIES} attempts: {last_exc}")
    rec = {**task, "status": "failed", "error": str(last_exc),
           "ts": datetime.utcnow().isoformat()}
    with open(req_log, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download ECMWF S2S reforecasts for climatology (90-day, 6 vars)."
    )
    parser.add_argument("--date",       type=str, default=None,
                        help="Single init date YYYY-MM-DD for testing")
    parser.add_argument("--year-start", type=int, default=YEAR_START,
                        help=f"First year (default {YEAR_START})")
    parser.add_argument("--year-end",   type=int, default=YEAR_END,
                        help=f"Last year (default {YEAR_END})")
    parser.add_argument("--no-pf",     action="store_true",
                        help="Control forecast only (skip perturbed members)")
    parser.add_argument("--sfc-only",  action="store_true",
                        help="Surface variables only (tp, 2t, mx2t, mn2t, msl)")
    parser.add_argument("--pl-only",   action="store_true",
                        help="Pressure-level only (Z500)")
    parser.add_argument("--workers",   type=int, default=4,
                        help="Parallel download threads (default 4)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print all tasks, download nothing")
    args = parser.parse_args()

    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log, log_file = setup_logging(run_id)
    req_log = LOG_DIR / "requests.jsonl"

    # Build date list
    if args.date:
        dates = [pd.Timestamp(args.date)]
        if dates[0].day not in REFORECAST_DAYS:
            log.warning(
                f"{args.date} is not a CY49R1+ reforecast day "
                f"(valid: {sorted(REFORECAST_DAYS)})"
            )
    else:
        dates = generate_init_dates(args.year_start, args.year_end)

    ftypes = ["cf"] if args.no_pf else ["cf", "pf"]
    do_sfc = not args.pl_only
    do_pl  = not args.sfc_only

    tasks   = build_tasks(dates, ftypes, do_sfc, do_pl)
    pending = [t for t in tasks if not is_done(Path(t["out"]))]
    n_done  = len(tasks) - len(pending)

    log.info("=" * 70)
    log.info("ECMWF S2S Reforecast Download")
    log.info(f"  Data dir    : {DATA_DIR}")
    log.info(f"  Log dir     : {LOG_DIR}")
    log.info(f"  Log file    : {log_file}")
    log.info(f"  Init dates  : {len(dates)}  "
             f"({dates[0].date()} → {dates[-1].date()})")
    log.info(f"  Schedule    : days {sorted(REFORECAST_DAYS)} of each month (CY49R1+)")
    log.info(f"  Fcst types  : {ftypes}")
    log.info(f"  Lead time   : 90 days  (steps 24–2160 h)")
    log.info(f"  Sfc vars    : {list(SURFACE_VARS.values()) if do_sfc else 'skipped'}")
    log.info(f"  PL vars     : z @ 500 hPa" if do_pl else "  PL vars     : skipped")
    log.info(f"  Tasks total : {len(tasks)}")
    log.info(f"  Already done: {n_done}")
    log.info(f"  Pending     : {len(pending)}")
    log.info(f"  Workers     : {args.workers}")
    log.info("=" * 70)

    if args.dry_run:
        log.info("DRY RUN — listing all tasks:")
        for t in tasks:
            tag = "EXISTS" if is_done(Path(t["out"])) else "PENDING"
            log.info(f"  [{tag}]  {t['label']}")
        log.info("=" * 70)
        log.info(f"Dry-run complete.  {len(pending)} files would be downloaded.")
        return

    if not pending:
        log.info("Nothing to download — all files already exist.")
        return

    # ── Run downloads ─────────────────────────────────────────────────────────
    counts = {"success": 0, "skipped": n_done, "failed": 0}
    failed = []
    done_n = n_done

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, t, log, req_log): t for t in pending}
        for fut in as_completed(futures):
            done_n += 1
            try:
                rec = fut.result()
                counts[rec["status"]] += 1
                if rec["status"] == "failed":
                    failed.append(rec["label"])
            except Exception as exc:
                counts["failed"] += 1
                log.error(f"Worker exception: {exc}")
            pct = 100 * done_n / len(tasks)
            log.info(
                f"Progress {done_n}/{len(tasks)} ({pct:.1f}%)  "
                f"ok={counts['success']}  skip={counts['skipped']}  "
                f"fail={counts['failed']}"
            )

    log.info("=" * 70)
    log.info("DOWNLOAD COMPLETE")
    log.info(f"  Success : {counts['success']}")
    log.info(f"  Skipped : {counts['skipped']}")
    log.info(f"  Failed  : {counts['failed']}")
    if failed:
        log.info("  Failed files:")
        for f in failed:
            log.info(f"    {f}")
    log.info(f"  Log     : {log_file}")
    log.info(f"  Req log : {req_log}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
