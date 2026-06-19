# S2S Data Pipeline

Download and preprocess S2S (subseasonal-to-seasonal) reforecast/hindcast data from ECMWF, FuXi, and Spire, and compute climatologies for verification.

## Data sources

| Model | Source | Period | Size |
|-------|--------|--------|------|
| ECMWF | CDS API (`s2s-reforecasts`) | 2000–2019 | ~few hundred GB |
| FuXi | HuggingFace `FudanFuXi/FuXi-S2S` | 2002–2021 | ~6.5 TB |
| Spire | Arraylake `artamta/s2s-research` | — | zarr store |

## Output layout

```
/storage/raj.ayush/All_Model_Data/
├── models/
│   ├── ecmwf/
│   │   ├── data/        # tp_cf_MMDD.grib, z500_cf_MMDD.grib, ...
│   │   └── clima/       # tp_clima_MMDD.nc, z500_clima_MMDD.nc
│   ├── fuxi/
│   │   ├── data/        # YYYYMMDD.7z archives
│   │   └── clima/       # tp_clima_MMDD.nc, z500_clima_MMDD.nc, t2m_clima_MMDD.nc
│   ├── ncep/
│   │   ├── data/
│   │   └── clima/
│   └── spire/
│       ├── data/        # s2s-research.zarr
│       └── clima/
└── ecmwf/
    └── reforecasts/     # by variable: 2t, msl, tp, u, v, z, q, t, mn2t, mx2t
```

## Setup

```bash
conda activate s2s-hind
pip install -r requirements.txt
```

You also need:
- `~/.cdsapirc` configured for ECMWF CDS API access
- `~/.arraylake/` configured for Spire Arraylake access
- HuggingFace login for FuXi (`huggingface-cli login`)

## Usage

### Download

Run each script inside a `tmux` session so it survives disconnection:

```bash
# ECMWF reforecasts (tp + Z500, 2000-2019, all Mon+Thu init dates)
tmux new -s ecmwf_download
python download/ecmwf_reforecast.py

# FuXi hindcasts (~6.5 TB from HuggingFace, resumable)
tmux new -s fuxi_download
python download/fuxi_reforecast.py

# Spire hindcast (zarr from Arraylake, resumable)
tmux new -s spire_download
conda activate s2s-hind
python download/spire_hindcast.py
```

### Compute climatology

```bash
# ECMWF climatology (single SLURM job)
sbatch slurm/ecmwf_clima.sbatch

# FuXi climatology (SLURM array: 7 parallel tasks)
sbatch slurm/fuxi_clima.sbatch
```

Or run directly without SLURM:

```bash
python climatology/ecmwf_clima.py
python climatology/fuxi_clima.py
```

## Variables

| Variable | Description |
|----------|-------------|
| `tp` | Total precipitation |
| `z500` / `gh` | Geopotential height at 500 hPa |
| `t2m` | 2-metre temperature |
| `msl` | Mean sea level pressure |
| `u`, `v` | Wind components |

## Notes

- All scripts are **resumable** — they skip files that already exist and have non-zero size.
- ECMWF init dates use 2020 as the base year to generate Mon+Thu schedule; actual reforecast data covers 2000–2019.
- FuXi climatology uses 51 ensemble members × ~20 years per MMDD.
- SLURM jobs use the `s2s-hind` conda env at `/home/raj.ayush/.conda/envs/s2s-hind/`.
