#!/usr/bin/env python3
"""
compute_ecmwf_clima.py
======================
Computes ECMWF S2S reforecast climatology (mean + std) from
control + perturbed reforecast grib files.

For each init MMDD:
  - Loads tp_cf + tp_pf  (control + 10 perturbed members, 20 years)
  - Loads z500_cf + z500_pf
  - Computes mean and std over all members x years
  - Saves to clima/ as NetCDF

Output:
  /storage/raj.ayush/All_Model_Data/models/ecmwf/clima/
      tp_clima_MMDD.nc     shape=(step=46, lat=34, lon=34)
      z500_clima_MMDD.nc
"""

import sys
from pathlib import Path
import numpy as np
import xarray as xr
import cfgrib

DATA_DIR  = Path("/storage/raj.ayush/All_Model_Data/models/ecmwf/data")
CLIMA_DIR = Path("/storage/raj.ayush/All_Model_Data/models/ecmwf/clima")
CLIMA_DIR.mkdir(parents=True, exist_ok=True)

# All MMDD init dates (derived from filenames)
all_mmdds = sorted(set(
    f.stem.split("_")[-1]
    for f in DATA_DIR.glob("tp_cf_*.grib")
))
print(f"Found {len(all_mmdds)} init dates to process.")

def load_and_stack(cf_file, pf_file, varname):
    """Load control + perturbed grib, return stacked DataArray (member*year, step, lat, lon)."""
    arrays = []

    # Control forecast: shape (time=20, step, lat, lon)
    if cf_file.exists():
        ds_cf = cfgrib.open_datasets(str(cf_file))
        for ds in ds_cf:
            if varname in ds:
                da = ds[varname]  # (time, step, lat, lon)
                # Rename time -> member_year, stack as extra samples
                da = da.rename({"time": "sample"})
                arrays.append(da)
                break

    # Perturbed forecast: shape (number=10, time=20, step, lat, lon)
    if pf_file.exists():
        ds_pf = cfgrib.open_datasets(str(pf_file))
        for ds in ds_pf:
            if varname in ds:
                da = ds[varname]  # (number, time, step, lat, lon)
                # Stack number x time into single sample dim, drop coords to allow concat
                da = da.stack(sample=("number", "time")).transpose("sample", "step", "latitude", "longitude")
                da = da.reset_index("sample").drop_vars(["number", "time"], errors="ignore")
                arrays.append(da)
                break

    if not arrays:
        return None

    # Convert all to plain numpy-backed DataArrays to avoid index conflicts
    clean = []
    for a in arrays:
        vals = a.values
        coords = {
            "step":      a.coords["step"].values,
            "latitude":  a.coords["latitude"].values,
            "longitude": a.coords["longitude"].values,
        }
        clean.append(xr.DataArray(vals, dims=["sample", "step", "latitude", "longitude"],
                                  coords={k: (d, v) for (k, v), d in
                                          zip(coords.items(), ["step", "latitude", "longitude"])}))
    combined = xr.concat(clean, dim="sample")
    return combined


for i, mmdd in enumerate(all_mmdds):
    print(f"\n[{i+1}/{len(all_mmdds)}] Processing MMDD={mmdd} ...", flush=True)

    for varname, label in [("tp", "tp"), ("gh", "z500")]:
        out_file = CLIMA_DIR / f"{label}_clima_{mmdd}.nc"
        if out_file.exists():
            print(f"  [SKIP] {out_file.name}")
            continue

        cf_file = DATA_DIR / f"{label}_cf_{mmdd}.grib" if label == "z500" else DATA_DIR / f"tp_cf_{mmdd}.grib"
        pf_file = DATA_DIR / f"{label}_pf_{mmdd}.grib" if label == "z500" else DATA_DIR / f"tp_pf_{mmdd}.grib"

        # Fix filenames for z500
        if label == "z500":
            cf_file = DATA_DIR / f"z500_cf_{mmdd}.grib"
            pf_file = DATA_DIR / f"z500_pf_{mmdd}.grib"

        da = load_and_stack(cf_file, pf_file, varname if varname == "tp" else "gh")
        if da is None:
            print(f"  [SKIP] {label} — no data found")
            continue

        # Compute mean and std over all samples (members x years)
        clim_mean = da.mean(dim="sample")
        clim_std  = da.std(dim="sample")

        ds_out = xr.Dataset({
            f"{label}_mean": clim_mean,
            f"{label}_std" : clim_std,
        })
        ds_out.attrs["description"] = f"ECMWF S2S reforecast climatology, init MMDD={mmdd}, 2000-2019, cf+pf"
        ds_out.to_netcdf(out_file)
        print(f"  [OK]  {out_file.name}")

print("\nDone.")
