#!/bin/bash
#SBATCH --job-name=ecmwf_jfm2026
#SBATCH --output=/storage/raj.ayush/s2s-data-pipeline/logs/ecmwf/jfm2026/slurm_%j.log
#SBATCH --error=/storage/raj.ayush/s2s-data-pipeline/logs/ecmwf/jfm2026/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00

PYTHON=/home/raj.ayush/.conda/envs/s2s-hind/bin/python
SCRIPT=/storage/raj.ayush/s2s-data-pipeline/download/ecmwf_forecast_jfm2026.py

mkdir -p /storage/raj.ayush/s2s-data-pipeline/logs/ecmwf/jfm2026

echo "================================================================"
echo "Job         : ECMWF S2S Real-Time Forecast Download  (JFM 2026)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "Start time  : $(date)"
echo "Script      : $SCRIPT"
echo "Output dir  : /storage/raj.ayush/All_Model_Data/ecmwf/jfm2026/"
echo "Log dir     : /storage/raj.ayush/s2s-data-pipeline/logs/ecmwf/jfm2026/"
echo "================================================================"

$PYTHON "$SCRIPT" --workers 4

echo "================================================================"
echo "End time : $(date)"
echo "================================================================"
