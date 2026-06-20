#!/bin/bash
#SBATCH --job-name=ecmwf_dl
#SBATCH --output=/storage/raj.ayush/s2s-data-pipeline/logs/ecmwf/slurm_%j.log
#SBATCH --error=/storage/raj.ayush/s2s-data-pipeline/logs/ecmwf/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00

# ── Environment ───────────────────────────────────────────────────────────────
PYTHON=/home/raj.ayush/.conda/envs/s2s-hind/bin/python
SCRIPT=/storage/raj.ayush/s2s-data-pipeline/download/ecmwf_reforecast.py
LOG_DIR=/storage/raj.ayush/s2s-data-pipeline/logs/ecmwf

mkdir -p "$LOG_DIR"

echo "================================================================"
echo "ECMWF S2S Reforecast Download Job"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "Start    : $(date)"
echo "Python   : $PYTHON"
echo "Script   : $SCRIPT"
echo "Log dir  : $LOG_DIR"
echo "================================================================"

# ── Run ───────────────────────────────────────────────────────────────────────
# 4 parallel workers (conservative — ECDS API throttles heavy concurrency)
$PYTHON "$SCRIPT" --workers 4

echo "================================================================"
echo "Done : $(date)"
echo "================================================================"
