#!/usr/bin/env python3
"""
download_fuxi_reforecast.py
===========================
Downloads all FuXi-S2S reforecast archives from HuggingFace.
Dataset : FudanFuXi/FuXi-S2S
Period  : 2002-2021 (2082 archives, ~6.5 TB total)
Output  : /storage/raj.ayush/All_Model_Data/models/fuxi/data/

Run inside tmux:
  tmux new -s fuxi_download
  cd /storage/raj.ayush/All_Model_Data
  python download_scripts/download_fuxi_reforecast.py
  Ctrl+B D to detach
"""

import time
from pathlib import Path
from huggingface_hub import list_repo_files, hf_hub_download

OUT_DIR = Path("/storage/raj.ayush/All_Model_Data/models/fuxi/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPO_ID = "FudanFuXi/FuXi-S2S"

# Get all .7z filenames from the repo
print("Fetching file list from HuggingFace ...", flush=True)
all_files = sorted([
    f for f in list_repo_files(REPO_ID, repo_type="dataset")
    if f.endswith(".7z")
])
total = len(all_files)
print(f"Found {total} archives to download.\n")

# ── MAIN LOOP ────────────────────────────────────────────────────────────────
t_start   = time.time()
done      = 0
skipped   = 0
failed    = []

for i, filename in enumerate(all_files):
    outfile = OUT_DIR / filename

    if outfile.exists() and outfile.stat().st_size > 0:
        skipped += 1
        print(f"[{i+1}/{total}] SKIP  {filename}  ({outfile.stat().st_size/1024**3:.2f} GB)")
        continue

    print(f"[{i+1}/{total}] GET   {filename} ...", flush=True)
    t0 = time.time()
    max_retries = 5
    success = False
    for attempt in range(1, max_retries + 1):
        try:
            # Remove partial file before retrying
            if outfile.exists() and outfile.stat().st_size == 0:
                outfile.unlink()
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=str(OUT_DIR),
            )
            elapsed  = time.time() - t0
            size_gb  = outfile.stat().st_size / 1024**3
            speed    = size_gb * 1024 / elapsed
            done    += 1
            success  = True

            elapsed_total = time.time() - t_start
            rate          = done / elapsed_total
            remaining     = (total - i - 1) / rate / 3600 if rate > 0 else 0

            print(f"         OK    {size_gb:.2f} GB  {speed:.0f} MB/s  | "
                  f"done={done}  skip={skipped}  fail={len(failed)}  ETA={remaining:.1f}h")
            break
        except Exception as e:
            print(f"         RETRY [{attempt}/{max_retries}]  {e}", flush=True)
            time.sleep(10 * attempt)  # back-off: 10s, 20s, 30s, 40s, 50s

    if not success:
        failed.append(filename)
        print(f"         FAIL  {filename} after {max_retries} attempts")

# ── SUMMARY ──────────────────────────────────────────────────────────────────
total_elapsed = (time.time() - t_start) / 3600
files         = list(OUT_DIR.glob("*.7z"))
total_gb      = sum(f.stat().st_size for f in files) / 1024**3

print("\n" + "=" * 65)
print("DOWNLOAD COMPLETE")
print(f"  Downloaded : {done}")
print(f"  Skipped    : {skipped}")
print(f"  Failed     : {len(failed)}")
print(f"  Total size : {total_gb:.1f} GB")
print(f"  Total time : {total_elapsed:.1f} hours")
if failed:
    print(f"\nFailed files:")
    for f in failed:
        print(f"  {f}")
print("=" * 65)
