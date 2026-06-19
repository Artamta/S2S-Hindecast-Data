#!/bin/bash
#SBATCH --job-name=ecmwf_clima
#SBATCH --output=/storage/raj.ayush/All_Model_Data/download_scripts/ecmwf_clima_%j.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00

echo "Starting ECMWF climatology computation..."
echo "Node: $(hostname)"
echo "Time: $(date)"

/home/raj.ayush/.conda/envs/s2s-hind/bin/python /storage/raj.ayush/All_Model_Data/download_scripts/compute_ecmwf_clima.py

echo "Done: $(date)"
