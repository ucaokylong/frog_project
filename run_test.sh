#!/bin/bash
#SBATCH -p GPU-1
#SBATCH -J FROG_LFE_PPN_TESTING
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH -o FROG_LFE_Output-%j.log

cd $SLURM_SUBMIT_DIR

# ==========================================
# BƯỚC 0: DỌN DẸP "BÓNG MA" MÔI TRƯỜNG CŨ
# ==========================================
unset PYTHONPATH
unset PYTHONHOME

# ==========================================
# BƯỚC 1: KÍCH HOẠT MÔI TRƯỜNG CHUẨN
# ==========================================
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate mamba_venv

# Ép hệ thống ưu tiên tuyệt đối Python của env
export PATH=$HOME/miniconda3/envs/mamba_venv/bin:$PATH

# ==========================================
# BƯỚC 2: THIẾT LẬP CUDA RUNTIME
# ==========================================
module load cuda/11.8
export CUDA_HOME=$CUDA_ROOT
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "[*] Thoi gian bat dau: $(date)"
echo "[*] Python dang dung: $(which python)"

# ==========================================
# BƯỚC 3: PIPELINE LFE-PPN (TEST)
# ==========================================

echo "=========================================================="
echo "[STEP 1] TESTING LFE Segmentation (Basic)..."
echo "=========================================================="

python -u test_basic.py

echo "[SUCCESS] TESTING LFE-PPN pipeline hoan thanh luc: $(date)"