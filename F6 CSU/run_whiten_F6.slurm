#!/bin/bash
#SBATCH --job-name=whiten_F6
#SBATCH --account=pas1266
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mairajuddin.1@osu.edu
#SBATCH --output=logs/whiten_F6_%j.out
#SBATCH --error=logs/whiten_F6_%j.err
# =============================================================================
# Whiten the CSU innovations, then re-run F6 on the clean single-tier stream.
# Needs, in this directory: cps_realdata.py, cps_experiments.py,
# cps_companion.py, cps_realdata_whiten.py, cps_companion_F6.py, and the R0
# cache (results/R0_csu_cache.npz).  CPU-only, default queue (no --partition).
# =============================================================================
module load miniconda3/24.1.2-py310
conda activate tft_env_torch

cd $SLURM_SUBMIT_DIR
mkdir -p logs results

# ---- Step 1: whiten -----------------------------------------------------
# refit: sweep VARX order, pick the whitest (lowest NOATTACK baseline_maxr).
# Raise the ceiling if p*=120 is still selected (baseline not yet flat).
# If the R0 cache is unavailable, fall back to re-tiering the existing file:
#python cps_realdata_whiten.py --mode from_existing --tier T2_varx --out_dir results
#	python cps_realdata_whiten.py --mode refit --orders 40,60,80,100,120 --out_dir results

# ---- Step 2: F6 on the clean, whitened, single-tier innovations ---------
#python cps_companion_F6.py \
    --innov_file results/R2_innovations_white.npz \
    --out_dir results --white_thresh 0.05
python cps_companion_F6.py --innov_file results/R2_innovations.npz --out_dir results --white_thresh 0.05
echo "done  $(date)"
