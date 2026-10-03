#!/bin/bash
#SBATCH --job-name=companion
#SBATCH --account=pas1266
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mairajuddin.1@osu.edu
#SBATCH --output=logs/companion_%A_%a.out
#SBATCH --error=logs/companion_%A_%a.err
#SBATCH --array=0-4
# =============================================================================
# Companion experiments F1..F5.  CPU-only (no torch), default queue.
# NO --partition line, per the note that a serial partition caused problems.
# Env setup mirrors the working cps_experiments.py job.
# F6 (real CAN data) is a separate job -- see run_companion_F6.slurm.
# =============================================================================
module load miniconda3/24.1.2-py310
conda activate tft_env_torch

cd $SLURM_SUBMIT_DIR
mkdir -p logs results

echo "host=$(hostname)  task=${SLURM_ARRAY_TASK_ID}  $(date)"
# ARRAY_ORDER inside cps_companion.py: 0=F4, 1=F2, 2=F3, 3=F1, 4=F5
python cps_companion.py --exp "array:${SLURM_ARRAY_TASK_ID}" --out_dir results

echo "done task=${SLURM_ARRAY_TASK_ID}  $(date)"
