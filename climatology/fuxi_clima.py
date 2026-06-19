#!/usr/bin/env python3
"""
compute_fuxi_clima.py
=====================
Compute FuXi S2S hindcast climatology (mean + std) per init MMDD.

Uses multiprocessing to extract and load archives in parallel.
Each worker handles one archive (one year): extracts to /tmp, loads
all 51 members x 42 steps for z500/t2m/tp, returns numpy arrays, cleans up.

Output:
  /storage/raj.ayush/All_Model_Data/models/fuxi/clima/
      tp_clima_MMDD.nc     shape=(step=42, lat=121, lon=240)
      z500_clima_MMDD.nc
      t2m_clima_MMDD.nc
"""

import os
import shutil
import tempfile
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict

import numpy as np
import xarray as xr
import py7zr

DATA_DIR  = Path("/storage/raj.ayush/All_Model_Data/models/fuxi/data")
CLIMA_DIR = Path("/storage/raj.ayush/All_Model_Data/models/fuxi/clima")
CLIMA_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ["z500", "t2m", "tp"]
MEMBERS  = [f"{m:02d}" for m in range(51)]
STEPS    = [f"{s:02d}" for s in range(1, 43)]
N_WORKERS = int(os.environ.get("FUXI_WORKERS", 20))


def process_archive(arch_path):
    """Extract one archive and return {var: (n_members, n_steps, lat, lon)}."""
    arch = Path(arch_path)
    tmpdir = Path(tempfile.mkdtemp(prefix="fuxi_"))
    try:
        with py7zr.SevenZipFile(arch, mode="r") as z:
            z.extractall(path=tmpdir)

        date_str = arch.stem
        year     = date_str[:4]
        # Some archives nest as YYYY/YYYYMMDD/member, others as YYYYMMDD/member
        base = tmpdir / year / date_str / "member"
        if not base.exists():
            base = tmpdir / date_str / "member"
        if not base.exists():
            print(f"  [WARN] {arch.name}: member dir not found under {tmpdir}", flush=True)
            return None

        lat_vals = lon_vals = None
        member_chunks = {v: [] for v in CHANNELS}

        for mem in MEMBERS:
            mem_dir = base / mem
            if not mem_dir.exists():
                continue
            step_arrays = {v: [] for v in CHANNELS}
            for step in STEPS:
                nc_path = mem_dir / f"{step}.nc"
                if not nc_path.exists():
                    continue
                ds = xr.open_dataset(nc_path)
                da = ds["__xarray_dataarray_variable__"].squeeze()
                ch_list = list(da.coords["channel"].values)
                if lat_vals is None:
                    lat_vals = da.coords["lat"].values if "lat" in da.coords else da.coords["latitude"].values
                    lon_vals = da.coords["lon"].values if "lon" in da.coords else da.coords["longitude"].values
                for var in CHANNELS:
                    idx = ch_list.index(var)
                    step_arrays[var].append(da.isel(channel=idx).values)
                ds.close()
            for var in CHANNELS:
                if step_arrays[var]:
                    member_chunks[var].append(np.stack(step_arrays[var], axis=0))

        result = {}
        for var in CHANNELS:
            if member_chunks[var]:
                result[var] = np.stack(member_chunks[var], axis=0)  # (n_members, n_steps, lat, lon)
        result["_lat"] = lat_vals
        result["_lon"] = lon_vals
        return result

    except Exception as e:
        print(f"  [WARN] {arch.name}: {e}", flush=True)
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    # Group archives by MMDD
    mmdd_to_archives = defaultdict(list)
    for f in sorted(DATA_DIR.glob("*.7z")):
        mmdd = f.stem[4:]
        mmdd_to_archives[mmdd].append(f)

    all_mmdds = sorted(mmdd_to_archives.keys())
    print(f"Found {len(all_mmdds)} unique MMDDs across {sum(len(v) for v in mmdd_to_archives.values())} archives.")

    done_mmdds = set()
    for mmdd in all_mmdds:
        if all((CLIMA_DIR / f"{var}_clima_{mmdd}.nc").exists() for var in CHANNELS):
            done_mmdds.add(mmdd)
    print(f"Already complete: {len(done_mmdds)} / {len(all_mmdds)} MMDDs")

    todo = [m for m in all_mmdds if m not in done_mmdds]
    print(f"To process: {len(todo)} MMDDs")

    # Job array: each SLURM task handles a chunk of MMDDs
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    n_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    chunk_size = -(-len(todo) // n_tasks)
    todo = todo[task_id * chunk_size : (task_id + 1) * chunk_size]
    print(f"Task {task_id}/{n_tasks-1}: {len(todo)} MMDDs  workers={N_WORKERS}\n")

    for i, mmdd in enumerate(todo):
        archives = mmdd_to_archives[mmdd]
        print(f"[{i+1}/{len(todo)}] MMDD={mmdd}  ({len(archives)} years, {N_WORKERS} parallel workers) ...", flush=True)

        # Check partial completion
        vars_needed = [v for v in CHANNELS if not (CLIMA_DIR / f"{v}_clima_{mmdd}.nc").exists()]
        if not vars_needed:
            print(f"  [SKIP] all vars done")
            continue

        # Parallel extraction
        with mp.Pool(processes=N_WORKERS) as pool:
            results = pool.map(process_archive, [str(a) for a in archives])

        # Filter failed
        results = [r for r in results if r is not None]
        if not results:
            print(f"  [WARN] no data for MMDD={mmdd}")
            continue

        # Get coords
        lat_vals = next((r["_lat"] for r in results if r.get("_lat") is not None), np.linspace(90, -90, 121))
        lon_vals = next((r["_lon"] for r in results if r.get("_lon") is not None), np.arange(0, 360, 1.5))
        step_coords = np.arange(1, 43)

        for var in vars_needed:
            out_file = CLIMA_DIR / f"{var}_clima_{mmdd}.nc"
            chunks = [r[var] for r in results if var in r]
            if not chunks:
                print(f"  [WARN] no data for {var}")
                continue

            all_data = np.concatenate(chunks, axis=0)  # (total_samples, n_steps, lat, lon)
            clim_mean = all_data.mean(axis=0)
            clim_std  = all_data.std(axis=0)
            n_steps = clim_mean.shape[0]

            ds_out = xr.Dataset({
                f"{var}_mean": xr.DataArray(clim_mean, dims=["step", "lat", "lon"],
                                            coords={"step": step_coords[:n_steps], "lat": lat_vals, "lon": lon_vals}),
                f"{var}_std":  xr.DataArray(clim_std,  dims=["step", "lat", "lon"],
                                            coords={"step": step_coords[:n_steps], "lat": lat_vals, "lon": lon_vals}),
            })
            ds_out.attrs["description"] = (
                f"FuXi-S2S hindcast climatology, init MMDD={mmdd}, "
                f"{len(archives)} years x 51 members"
            )
            ds_out.to_netcdf(out_file)
            print(f"  [OK]  {out_file.name}", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
