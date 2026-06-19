#!/bin/bash
#SBATCH --job-name=fuxi_clima
#SBATCH --output=/storage/raj.ayush/All_Model_Data/download_scripts/fuxi_clima_%A_%a.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-6

export CUDA_VISIBLE_DEVICES=0
export FUXI_WORKERS=20

echo "Task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} on $(hostname) — $(date)"

/home/raj.ayush/.conda/envs/s2s-hind/bin/python \
    /storage/raj.ayush/All_Model_Data/download_scripts/compute_fuxi_clima.py

echo "Done: $(date)"
