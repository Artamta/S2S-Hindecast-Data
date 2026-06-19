#!/usr/bin/env python3
"""
download_spire_hindcast.py
==========================
Downloads the full Spire S2S hindcast from arraylake to local zarr store.

Source  : arraylake repo artamta/s2s-research (branch: main)
Output  : /storage/raj.ayush/All_Model_Data/models/spire/data/s2s-research.zarr

Run inside tmux:
  tmux new -s spire_download
  conda activate s2s-hind
  cd /storage/raj.ayush/All_Model_Data
  python download_scripts/download_spire_hindcast.py
  Ctrl+B D to detach
"""

import time
from pathlib import Path
import zarr
import numpy as np
from arraylake import Client

OUT_PATH = Path("/storage/raj.ayush/All_Model_Data/models/spire/data/s2s-research.zarr")
OUT_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("Spire S2S Hindcast Download")
print(f"Source : arraylake artamta/s2s-research (main)")
print(f"Output : {OUT_PATH}")
print("=" * 65)

# ── Connect to arraylake ─────────────────────────────────────────────────────
print("\nConnecting to arraylake ...", flush=True)
client  = Client()
repo    = client.get_repo("artamta/s2s-research")
session = repo.readonly_session(branch="main")
src     = zarr.open_group(session.store, zarr_format=3, mode="r")

# ── Open/create local zarr store ─────────────────────────────────────────────
dst = zarr.open_group(str(OUT_PATH), zarr_format=3, mode="a")

# ── Copy all groups and arrays ───────────────────────────────────────────────
def copy_group(src_grp, dst_grp, path=""):
    for key in src_grp.keys():
        src_item = src_grp[key]
        item_path = f"{path}/{key}" if path else key

        if isinstance(src_item, zarr.Group):
            print(f"\n  Group: {item_path}/", flush=True)
            if key not in dst_grp:
                dst_grp.require_group(key)
            copy_group(src_item, dst_grp[key], item_path)

        else:
            # It's an array
            if key in dst_grp:
                existing = dst_grp[key]
                if existing.shape == src_item.shape:
                    print(f"  [SKIP] {item_path}  {src_item.shape}", flush=True)
                    continue
                else:
                    print(f"  [OVERWRITE] {item_path}  shape mismatch", flush=True)

            print(f"  [GET]  {item_path}  shape={src_item.shape}  dtype={src_item.dtype} ...", flush=True)
            t0 = time.time()
            max_retries = 5
            success = False
            for attempt in range(1, max_retries + 1):
                try:
                    data = src_item[:]
                    dst_grp.create_array(
                        key,
                        data=data,
                        chunks=src_item.chunks if hasattr(src_item, 'chunks') else True,
                        overwrite=True,
                    )
                    elapsed = time.time() - t0
                    mb = data.nbytes / 1024**2
                    print(f"         OK  {mb:.1f} MB  {elapsed:.1f}s", flush=True)
                    success = True
                    break
                except Exception as e:
                    print(f"         RETRY [{attempt}/{max_retries}]: {e}", flush=True)
                    time.sleep(10 * attempt)
            if not success:
                print(f"         FAIL after {max_retries} attempts: {item_path}", flush=True)

t_start = time.time()
copy_group(src, dst)

total_elapsed = (time.time() - t_start) / 60
print("\n" + "=" * 65)
print(f"Done. Total time: {total_elapsed:.1f} minutes")
print(f"Output: {OUT_PATH}")
print("=" * 65)
