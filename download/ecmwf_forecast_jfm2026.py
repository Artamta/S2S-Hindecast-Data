#!/usr/bin/env python3
"""
ecmwf_forecast_jfm2026.py
==========================
Download ECMWF S2S real-time operational forecasts for JFM 2026
(January – March 2026, all available init dates).

This is the ACTUAL forecast data used for verification — NOT hindcasts.

Ensemble size (real-time forecasts, CY49R1+):
  101 members total — 1 control (cf) + 100 perturbed (pf)

Init dates (CY49R1+, daily since 12 Nov 2024):
  Every day in JFM 2026: 2026-01-01 → 2026-03-31  (90 dates)

Lead time : 90 days  (steps 24, 48, … 2160 h)
Dataset   : s2s-forecasts  (NOT s2s-reforecasts)

Variables (6 — same as reforecast script):
  Surface  : tp, 2t, mx2t, mn2t, msl
  Pressure : z @ 500 hPa (Z500)

Output
------
  /storage/raj.ayush/All_Model_Data/ecmwf/jfm2026/
  ├── tp/        YYYYMMDD_cf.nc   YYYYMMDD_pf.nc
  ├── 2t/        ...
  ├── mx2t/      ...
  ├── mn2t/      ...
  ├── msl/       ...
  └── z/500/     YYYYMMDD_cf.nc   YYYYMMDD_pf.nc

  Each file: one init date × one variable × one fcst type
             cf shape: (lead_day=90, lat, lon)
             pf shape: (member=100, lead_day=90, lat, lon)

Logs
----
  /storage/raj.ayush/s2s-data-pipeline/logs/ecmwf/jfm2026/
  ├── run_YYYYMMDD_HHMMSS.log
  └── requests.jsonl

Usage
-----
  # Dry-run on login node (do this first)
  python download/ecmwf_forecast_jfm2026.py --dry-run

  # Single date test
  python download/ecmwf_forecast_jfm2026.py --date 2026-01-01

  # Full run (via SLURM — see slurm/ecmwf_jfm2026.sbatch)
  python download/ecmwf_forecast_jfm2026.py --workers 4

  # Surface only
  python download/ecmwf_forecast_jfm2026.py --sfc-only

  # Control only
  python download/ecmwf_forecast_jfm2026.py --no-pf

Requirements
------------
  conda activate s2s-hind
  ~/.cdsapirc: url: https://ecds.ecmwf.int/api  /  key: <your-key>
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
DATA_DIR = Path("/storage/raj.ayush/All_Model_Data/ecmwf/jfm2026")
LOG_DIR  = Path("/storage/raj.ayush/s2s-data-pipeline/logs/ecmwf/jfm2026")

# ── FORECAST PERIOD ───────────────────────────────────────────────────────────
# All daily init dates in JFM 2026
DATE_START = "2026-01-01"
DATE_END   = "2026-03-31"

# ── FORECAST CONFIGURATION ────────────────────────────────────────────────────
# 90-day lead time at daily resolution
STEPS = [str(h) for h in range(24, 90 * 24 + 1, 24)]   # 24 … 2160

# 6 paper variables
SURFACE_VARS = {
    "total_precipitation":                          "tp",
    "2m_temperature":                               "2t",
    "maximum_2m_temperature_in_the_last_24_hours":  "mx2t",
    "minimum_2m_temperature_in_the_last_24_hours":  "mn2t",
    "mean_sea_level_pressure":                      "msl",
}
PL_VARS   = {"geopotential": "z"}
PL_LEVELS = ["500"]

# Forecast types (real-time uses control_forecast / perturbed_forecast)
FTYPES = {
    "cf": "control_forecast",
    "pf": "perturbed_forecast",
}

# ── RETRY ─────────────────────────────────────────────────────────────────────
MAX_RETRIES   = 5
RETRY_BACKOFF = [30, 60, 120, 300, 600]


# ── HELPERS ───────────────────────────────────────────────────────────────────
def setup_logging() -> tuple[logging.Logger, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__), log_file


def outpath(var: str, date: pd.Timestamp, ftype: str, level: str | None = None) -> Path:
    d = DATA_DIR / var / level if level else DATA_DIR / var
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{date:%Y%m%d}_{ftype}.nc"


def is_done(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def build_tasks(dates: list[pd.Timestamp], ftypes: list[str],
                do_sfc: bool, do_pl: bool) -> list[dict]:
    tasks = []
    for date in dates:
        for ftype in ftypes:
            base = {
                "origin":        "ecmwf",
                "forecast_type": FTYPES[ftype],
                "year":          str(date.year),
                "month":         f"{date.month:02d}",
                "day":           f"{date.day:02d}",
                "time":          "00:00",
                "step":          STEPS,
                "data_format":   "netcdf",
            }
            if do_sfc:
                for cds_name, short in SURFACE_VARS.items():
                    p = outpath(short, date, ftype)
                    tasks.append({"out": str(p), "label": f"{short}/{date:%Y%m%d}_{ftype}.nc",
                                  "req": {**base, "level_type": "single_level", "variable": cds_name}})
            if do_pl:
                for cds_name, short in PL_VARS.items():
                    for lev in PL_LEVELS:
                        p = outpath(short, date, ftype, lev)
                        tasks.append({"out": str(p), "label": f"{short}/{lev}/{date:%Y%m%d}_{ftype}.nc",
                                      "req": {**base, "level_type": "pressure_level",
                                              "variable": cds_name, "level": lev}})
    return tasks


def download_one(task: dict, log: logging.Logger, req_log: Path) -> dict:
    out = Path(task["out"])
    if is_done(out):
        log.info(f"SKIP   {task['label']}")
        return {**task, "status": "skipped"}

    client, last_exc = cdsapi.Client(quiet=True), None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(f"START  {task['label']}  (attempt {attempt}/{MAX_RETRIES})")
            t0 = time.time()
            client.retrieve("s2s-forecasts", task["req"], str(out))
            elapsed = time.time() - t0
            mb = out.stat().st_size / 1024 ** 2
            log.info(f"DONE   {task['label']}  {mb:.1f} MB  {elapsed:.0f}s")
            rec = {**task, "status": "success", "size_mb": round(mb, 2),
                   "elapsed_s": round(elapsed, 1), "ts": datetime.utcnow().isoformat()}
            with open(req_log, "a") as f:
                f.write(json.dumps(rec) + "\n")
            return rec
        except Exception as exc:
            last_exc = exc
            log.warning(f"RETRY  {task['label']}  attempt {attempt}: {exc}")
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF[attempt - 1]
                log.info(f"       waiting {wait}s …")
                time.sleep(wait)

    log.error(f"FAIL   {task['label']} after {MAX_RETRIES} attempts: {last_exc}")
    rec = {**task, "status": "failed", "error": str(last_exc),
           "ts": datetime.utcnow().isoformat()}
    with open(req_log, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def main():
    parser = argparse.ArgumentParser(
        description="Download ECMWF S2S real-time forecasts JFM 2026 (101 members, 90-day, 6 vars)."
    )
    parser.add_argument("--date",      type=str, default=None,
                        help="Single init date YYYY-MM-DD for testing")
    parser.add_argument("--no-pf",    action="store_true", help="Control only (no perturbed)")
    parser.add_argument("--sfc-only", action="store_true", help="Surface vars only")
    parser.add_argument("--pl-only",  action="store_true", help="Pressure-level only (Z500)")
    parser.add_argument("--workers",  type=int, default=4, help="Parallel threads (default 4)")
    parser.add_argument("--dry-run",  action="store_true", help="Print tasks, download nothing")
    args = parser.parse_args()

    log, log_file = setup_logging()
    req_log = LOG_DIR / "requests.jsonl"

    if args.date:
        dates = [pd.Timestamp(args.date)]
    else:
        dates = list(pd.date_range(DATE_START, DATE_END, freq="D"))

    ftypes = ["cf"] if args.no_pf else ["cf", "pf"]
    do_sfc = not args.pl_only
    do_pl  = not args.sfc_only

    tasks   = build_tasks(dates, ftypes, do_sfc, do_pl)
    pending = [t for t in tasks if not is_done(Path(t["out"]))]
    n_done  = len(tasks) - len(pending)

    log.info("=" * 70)
    log.info("ECMWF S2S Real-Time Forecast Download  (JFM 2026)")
    log.info(f"  Data dir     : {DATA_DIR}")
    log.info(f"  Log file     : {log_file}")
    log.info(f"  Period       : {DATE_START} → {DATE_END}")
    log.info(f"  Init dates   : {len(dates)}  (daily, all days in JFM 2026)")
    log.info(f"  Members      : 101  (1 cf + 100 pf)  ← real-time forecast")
    log.info(f"  Lead time    : 90 days  (steps 24–2160 h, daily)")
    log.info(f"  Dataset      : s2s-forecasts  (NOT reforecasts)")
    log.info(f"  Fcst types   : {ftypes}")
    log.info(f"  Sfc vars     : {list(SURFACE_VARS.values()) if do_sfc else 'skipped'}")
    log.info(f"  PL vars      : z @ 500 hPa" if do_pl else "  PL vars      : skipped")
    log.info(f"  Total tasks  : {len(tasks)}")
    log.info(f"  Already done : {n_done}")
    log.info(f"  To download  : {len(pending)}")
    log.info(f"  Workers      : {args.workers}")
    log.info("=" * 70)

    if args.dry_run:
        for t in tasks:
            tag = "EXISTS " if is_done(Path(t["out"])) else "PENDING"
            log.info(f"  [{tag}]  {t['label']}")
        log.info(f"Dry-run complete — {len(pending)} files would be downloaded.")
        return

    if not pending:
        log.info("Nothing to download — all files exist.")
        return

    counts = {"success": 0, "skipped": n_done, "failed": 0}
    failed, done_n = [], n_done

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
            log.info(f"Progress {done_n}/{len(tasks)} ({100*done_n/len(tasks):.1f}%)  "
                     f"ok={counts['success']}  skip={counts['skipped']}  fail={counts['failed']}")

    log.info("=" * 70)
    log.info("DOWNLOAD COMPLETE")
    log.info(f"  Success : {counts['success']}")
    log.info(f"  Skipped : {counts['skipped']}")
    log.info(f"  Failed  : {counts['failed']}")
    for f in failed:
        log.info(f"    FAILED: {f}")
    log.info(f"  Req log : {req_log}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
