#!/bin/bash
#SBATCH --job-name=companion_F6
#SBATCH --account=pas1266
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mairajuddin.1@osu.edu
#SBATCH --output=logs/companion_F6_%j.out
#SBATCH --error=logs/companion_F6_%j.err
# =============================================================================
# Upgraded F6: exposure on real CSU CAN innovations.
#   - contiguous held-out split (no interleave decimation)
#   - per-deployment (L,f) tuning + single-config transfer test
#   - NOATTACK baseline, dither baseline, multi-lag diagnostics, detector power
# Needs cps_companion_F6.py, cps_companion.py, cps_experiments.py and the
# innovations file in this directory.  CPU-only, default queue.
# =============================================================================
module load miniconda3/24.1.2-py310
conda activate tft_env_torch

cd $SLURM_SUBMIT_DIR
mkdir -p logs results

INNOV_FILE="${INNOV_FILE:-R2_innovations.npz}"
echo "host=$(hostname)  innov=${INNOV_FILE}  $(date)"

# Full grid.  The SUMMARY block at the end of the log is what to paste back.
python cps_companion_F6.py --innov_file "${INNOV_FILE}" --out_dir results

echo "done  $(date)"
