#!/usr/bin/env python3
"""
download_ecmwf_reforecast.py
============================
Downloads ECMWF S2S reforecasts (hindcasts) for climatology computation.

Variables : total precipitation (tp) + geopotential height at 500 hPa (gh)
Region    : India domain (0-50N, 55-105E) at 1.5 deg
Years     : 2000-2019 (20 hindcast years, all come in one request per init date)
Init dates: All Mondays + Thursdays across a full year (real-time date = 2020)
Ftypes    : control_reforecast (cf) + perturbed_reforecast (pf)

Output layout:
  /storage/raj.ayush/All_Model_Data/models/ecmwf/data/
      tp_cf_MMDD.grib      <- control reforecast, all 20 years packed inside
      tp_pf_MMDD.grib
      z500_cf_MMDD.grib
      z500_pf_MMDD.grib

Run inside tmux:
  tmux new -s ecmwf_download
  python download_scripts/download_ecmwf_reforecast.py
  Ctrl+B D  to detach
"""

from pathlib import Path
import pandas as pd
import cdsapi

# ── CONFIG ───────────────────────────────────────────────────────────────────
OUT_DIR = Path("/storage/raj.ayush/All_Model_Data/models/ecmwf/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use 2020 as the "real-time" year to generate Mon+Thu init dates.
# The reforecast response always contains years 2000-2019 regardless.
all_days   = pd.date_range("2020-01-01", "2020-12-31", freq="D")
INIT_DATES = [d for d in all_days if d.weekday() in (0, 3)]  # Mon=0, Thu=3

# Lead steps: 24h to 46 days
STEPS = [str(h) for h in range(24, 1105, 24)]

# India domain
AREA = [50, 55, 0, 105]   # [N, W, S, E]
GRID = [1.5, 1.5]

client = cdsapi.Client()

# ── HELPERS ──────────────────────────────────────────────────────────────────
def download(variable, level_type, ftype, date, outfile, extra=None):
    if outfile.exists() and outfile.stat().st_size > 0:
        print(f"  [SKIP] {outfile.name}")
        return

    request = {
        "origin": "ecmwf",
        "forecast_type": ftype,
        "level_type": level_type,
        "variable": variable,
        "year": str(date.year),
        "month": f"{date.month:02d}",
        "day": f"{date.day:02d}",
        "time": "00:00",
        "step": STEPS,
        "area": AREA,
        "grid": GRID,
        "data_format": "grib",
    }
    if extra:
        request.update(extra)

    print(f"  [GET]  {outfile.name} ...", flush=True)
    try:
        client.retrieve("s2s-reforecasts", request, str(outfile))
        mb = outfile.stat().st_size / 1024**2
        print(f"  [OK]   {outfile.name}  ({mb:.1f} MB)")
    except Exception as e:
        print(f"  [ERR]  {outfile.name}: {e}")


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    total = len(INIT_DATES)
    print("=" * 65)
    print("ECMWF S2S Reforecast Download")
    print(f"Init dates : {total} (Mon + Thu, full year using 2020 as base)")
    print(f"Hindcast   : 2000-2019 (20 years per request)")
    print(f"Variables  : tp, Z500")
    print(f"Forecast   : control + perturbed reforecast")
    print(f"Output     : {OUT_DIR}")
    print("=" * 65)

    for i, date in enumerate(INIT_DATES):
        mmdd = date.strftime("%m%d")
        print(f"\n[{i+1}/{total}] Init: {date.strftime('%a %Y-%m-%d')}  (MMDD={mmdd})")

        # Total Precipitation — surface
        download("tp", "single_level", "control_reforecast",   date, OUT_DIR / f"tp_cf_{mmdd}.grib")
        download("tp", "single_level", "perturbed_reforecast", date, OUT_DIR / f"tp_pf_{mmdd}.grib")

        # Z500 — pressure level
        download("gh", "pressure",     "control_reforecast",   date, OUT_DIR / f"z500_cf_{mmdd}.grib", extra={"level": "500"})
        download("gh", "pressure",     "perturbed_reforecast", date, OUT_DIR / f"z500_pf_{mmdd}.grib", extra={"level": "500"})

    print("\n" + "=" * 65)
    print("All done.")
    files = list(OUT_DIR.glob("*.grib"))
    total_mb = sum(f.stat().st_size for f in files) / 1024**2
    print(f"Files in output dir : {len(files)}")
    print(f"Total size          : {total_mb:.1f} MB")
    print("=" * 65)


if __name__ == "__main__":
    main()
