#!/usr/bin/env python3
"""
ecmwf_reforecast.py
===================
Download ECMWF S2S reforecasts (hindcasts) from ECDS.

ECMWF S2S reforecast initialization frequency depends on the model cycle:

  - CY48R1 and earlier (before 12 Nov 2024): Mon + Thu only (twice weekly)
  - CY49R1+ (from 12 Nov 2024, current):     Every odd day of month
                                              (1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31)

This script targets the current CY49R1+ schedule: ~15 init dates/month,
covering all odd calendar days.  Each request returns all 20 hindcast years
(2000-2019) packed in a single file.

Variables
---------
Surface (single_level):
  tp   — total precipitation
  2t   — 2 m temperature
  mx2t — maximum 2 m temperature in last 24 h
  mn2t — minimum 2 m temperature in last 24 h
  msl  — mean sea level pressure

Pressure level:
  z, t, u, v, q  at 200 / 500 / 850 / 1000 hPa

Output layout
-------------
  <OUT_BASE>/
  ├── tp/   <YYYYMMDD>_cf.nc   <YYYYMMDD>_pf.nc  ...
  ├── 2t/   ...
  ├── mx2t/ ...
  ├── mn2t/ ...
  ├── msl/  ...
  ├── z/200/  z/500/  z/850/  z/1000/
  ├── t/200/  ...
  ├── u/200/  ...
  ├── v/200/  ...
  └── q/200/  ...

Each file: one init date × one variable (× one level for PL vars)
           × one forecast type (cf / pf), NetCDF format.

Logs → <OUT_BASE>/_logs/

Usage
-----
  # Dry-run: show what would be downloaded
  python download/ecmwf_reforecast.py --dry-run

  # All init dates, all variables (2000-2019)
  python download/ecmwf_reforecast.py

  # Single init date test
  python download/ecmwf_reforecast.py --date 2020-01-02

  # Surface variables only (faster)
  python download/ecmwf_reforecast.py --sfc-only

  # Pressure-level variables only
  python download/ecmwf_reforecast.py --pl-only

  # Skip perturbed members (cf only)
  python download/ecmwf_reforecast.py --no-pf

  # More parallel workers
  python download/ecmwf_reforecast.py --workers 6

  # Run in background via tmux
  tmux new -s ecmwf_dl
  python download/ecmwf_reforecast.py --workers 4
  Ctrl+B D  to detach

Requirements
------------
  pip install cdsapi pandas
  ~/.cdsapirc must contain your ECDS key (url: https://ecds.ecmwf.int/api)
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
OUT_BASE = Path("/storage/raj.ayush/All_Model_Data/ecmwf/reforecasts")
LOG_DIR  = OUT_BASE / "_logs"

# ── HINDCAST PERIOD ───────────────────────────────────────────────────────────
# ECMWF provides a 20-year hindcast window.  All Mon+Thu dates in this range
# are valid init dates; each request returns all 20 years stacked internally.
YEAR_START = 2000
YEAR_END   = 2019

# ── CDS REQUEST PARAMETERS ────────────────────────────────────────────────────
STEPS = [str(h) for h in range(24, 24 * 46 + 1, 24)]   # day 1–46

SURFACE_VARS = {
    "total_precipitation":                          "tp",
    "2m_temperature":                               "2t",
    "maximum_2m_temperature_in_the_last_24_hours":  "mx2t",
    "minimum_2m_temperature_in_the_last_24_hours":  "mn2t",
    "mean_sea_level_pressure":                      "msl",
}

PL_VARS = {
    "geopotential":          "z",
    "temperature":           "t",
    "u_component_of_wind":   "u",
    "v_component_of_wind":   "v",
    "specific_humidity":     "q",
}

PRESSURE_LEVELS = ["200", "500", "850", "1000"]

FTYPES = {
    "cf": "control_reforecast",
    "pf": "perturbed_reforecast",
}

# ── RETRY ─────────────────────────────────────────────────────────────────────
MAX_RETRIES   = 5
RETRY_BACKOFF = [30, 60, 120, 300, 600]


# ── HELPERS ───────────────────────────────────────────────────────────────────
def _setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"download_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger(__name__)
    log.info(f"Log file: {log_file}")
    return log


def _outpath(short_name: str, date: pd.Timestamp, ftype_key: str,
             level: str | None = None) -> Path:
    d = OUT_BASE / short_name / level if level else OUT_BASE / short_name
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{date:%Y%m%d}_{ftype_key}.nc"


def _done(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def generate_init_dates(year_start: int, year_end: int) -> list[pd.Timestamp]:
    """All odd-day-of-month dates in [year_start, year_end] (CY49R1+ schedule).
    Odd days: 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31 of each month.
    Feb 29 is excluded (leap year odd day that doesn't exist in most years).
    """
    dates = pd.date_range(f"{year_start}-01-01", f"{year_end}-12-31", freq="D")
    return [d for d in dates if d.day % 2 == 1]


def build_tasks(dates: list[pd.Timestamp], ftypes: list[str],
                do_sfc: bool, do_pl: bool) -> list[dict]:
    tasks = []
    for date in dates:
        for ftype_key in ftypes:
            base_req = {
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
                for cds_name, short_name in SURFACE_VARS.items():
                    tasks.append({
                        "out":   str(_outpath(short_name, date, ftype_key)),
                        "req":   {**base_req, "level_type": "single_level",
                                  "variable": cds_name},
                        "label": f"{short_name}/{date:%Y%m%d}_{ftype_key}.nc",
                    })
            if do_pl:
                for cds_name, short_name in PL_VARS.items():
                    for lev in PRESSURE_LEVELS:
                        tasks.append({
                            "out":   str(_outpath(short_name, date, ftype_key, lev)),
                            "req":   {**base_req, "level_type": "pressure_level",
                                      "variable": cds_name, "level": lev},
                            "label": f"{short_name}/{lev}/{date:%Y%m%d}_{ftype_key}.nc",
                        })
    return tasks


# ── DOWNLOAD ──────────────────────────────────────────────────────────────────
def download_one(task: dict, log: logging.Logger, req_log: Path) -> dict:
    out = Path(task["out"])
    if _done(out):
        log.info(f"SKIP  {task['label']}")
        return {**task, "status": "skipped"}

    last_exc = None
    client = cdsapi.Client(quiet=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(f"START {task['label']}  (attempt {attempt})")
            t0 = time.time()
            client.retrieve("s2s-reforecasts", task["req"], str(out))
            elapsed = time.time() - t0
            size_mb = out.stat().st_size / 1024 ** 2
            log.info(f"DONE  {task['label']}  {size_mb:.1f} MB  {elapsed:.0f}s")
            rec = {**task, "status": "success", "size_mb": round(size_mb, 2),
                   "elapsed_s": round(elapsed, 1), "ts": datetime.utcnow().isoformat()}
            with open(req_log, "a") as f:
                f.write(json.dumps(rec) + "\n")
            return rec
        except Exception as exc:
            last_exc = exc
            log.warning(f"FAIL  {task['label']}  attempt {attempt}: {exc}")
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF[attempt - 1]
                log.info(f"      retrying in {wait}s ...")
                time.sleep(wait)

    log.error(f"GIVE UP {task['label']}: {last_exc}")
    rec = {**task, "status": "failed", "error": str(last_exc),
           "ts": datetime.utcnow().isoformat()}
    with open(req_log, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download ECMWF S2S reforecasts — all Mon+Thu inits, 2000-2019."
    )
    parser.add_argument("--date",       type=str, default=None,
                        help="Single init date YYYY-MM-DD (must be Mon or Thu)")
    parser.add_argument("--year-start", type=int, default=YEAR_START)
    parser.add_argument("--year-end",   type=int, default=YEAR_END)
    parser.add_argument("--no-pf",     action="store_true",
                        help="Skip perturbed reforecast; control only")
    parser.add_argument("--sfc-only",  action="store_true",
                        help="Surface variables only")
    parser.add_argument("--pl-only",   action="store_true",
                        help="Pressure-level variables only")
    parser.add_argument("--workers",   type=int, default=4,
                        help="Parallel download threads (default 4)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Show pending tasks without downloading")
    args = parser.parse_args()

    log     = _setup_logging()
    req_log = LOG_DIR / "requests.jsonl"

    if args.date:
        dates = [pd.Timestamp(args.date)]
        if dates[0].day % 2 == 0:
            log.warning(f"{args.date} is an even day — CY49R1+ reforecasts are issued on odd days only.")
    else:
        dates = generate_init_dates(args.year_start, args.year_end)

    ftypes = ["cf"] if args.no_pf else ["cf", "pf"]
    do_sfc = not args.pl_only
    do_pl  = not args.sfc_only

    tasks   = build_tasks(dates, ftypes, do_sfc, do_pl)
    pending = [t for t in tasks if not _done(Path(t["out"]))]

    log.info("=" * 68)
    log.info("ECMWF S2S Reforecast Downloader")
    log.info(f"  Output      : {OUT_BASE}")
    log.info(f"  Init dates  : {len(dates)}  ({dates[0].date()} → {dates[-1].date()})")
    log.info(f"  NOTE: CY49R1+ reforecasts on odd days of month (1,3,5...31)")
    log.info(f"  Fcst types  : {ftypes}")
    log.info(f"  Sfc vars    : {list(SURFACE_VARS.values()) if do_sfc else 'skipped'}")
    log.info(f"  PL vars     : {list(PL_VARS.values())} @ {PRESSURE_LEVELS} hPa" if do_pl else "  PL vars     : skipped")
    log.info(f"  Total tasks : {len(tasks)}  |  done: {len(tasks)-len(pending)}  |  pending: {len(pending)}")
    log.info(f"  Workers     : {args.workers}")
    log.info("=" * 68)

    if args.dry_run:
        for t in tasks:
            tag = "EXISTS " if _done(Path(t["out"])) else "PENDING"
            print(f"  [{tag}] {t['label']}")
        log.info("Dry-run complete — nothing downloaded.")
        return

    if not pending:
        log.info("Nothing to download.")
        return

    counts = {"success": 0, "skipped": len(tasks) - len(pending), "failed": 0}
    failed = []
    done_n = counts["skipped"]

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
            log.info(f"Progress {done_n}/{len(tasks)} ({pct:.1f}%)  "
                     f"ok={counts['success']} skip={counts['skipped']} fail={counts['failed']}")

    log.info("=" * 68)
    log.info("DOWNLOAD COMPLETE")
    log.info(f"  Success  : {counts['success']}")
    log.info(f"  Skipped  : {counts['skipped']}")
    log.info(f"  Failed   : {counts['failed']}")
    for f in failed:
        log.info(f"    FAILED: {f}")
    log.info(f"  Req log  : {req_log}")
    log.info("=" * 68)


if __name__ == "__main__":
    main()
