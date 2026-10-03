#!/bin/bash
#SBATCH --job-name=companion_F6
#SBATCH --account=pas1266
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mairajuddin.1@osu.edu
#SBATCH --output=logs/companion_F6_%j.out
#SBATCH --error=logs/companion_F6_%j.err
# =============================================================================
# Companion experiment F6 -- exposure on real CSU CAN innovations.
# Needs VARX-recovered innovations as an .npz (key 'segments' = object array of
# 1-D innovation arrays, one per stationary segment) OR a CSV with columns
# segment,z.  Set INNOV_FILE before submitting:
#     INNOV_FILE=csu_innovations.npz sbatch run_companion_F6.slurm
# CPU-only, default queue, no --partition line.
# =============================================================================
module load miniconda3/24.1.2-py310
conda activate tft_env_torch

cd $SLURM_SUBMIT_DIR
mkdir -p logs results

INNOV_FILE="${INNOV_FILE:-csu_innovations.npz}"
echo "host=$(hostname)  innov=${INNOV_FILE}  $(date)"
python cps_companion.py --exp F6 --innov_file "${INNOV_FILE}" --out_dir results
echo "done  $(date)"
