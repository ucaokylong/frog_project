#!/bin/bash
#SBATCH -p GPU-1
#SBATCH -J FROG_DRSPAAM
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH -o FROG_DRSPAAM_Output-%j.log

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
# Vẫn dùng mamba_venv vì Torch 2.1 trong này đang chạy rất ổn định
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
# BƯỚC 3: PIPELINE DR-SPAAM (SEG -> LOC -> TEST)
# ==========================================

echo "=========================================================="
echo "[STEP 1] Training Temporal Segmentation (T=5)..."
echo "=========================================================="
python -u train_seg_temporal.py

# Kiểm tra nếu train Seg thành công thì mới chạy tiếp Loc
if [ -f "checkpoints/drspaam_seg_best.pth" ]; then
    echo "=========================================================="
    echo "[STEP 2] Training Temporal Localization (T=5)..."
    echo "=========================================================="
    python -u train_loc_temporal.py
    
    # Kiểm tra nếu train Loc thành công thì mới chạy Test
    if [ -f "checkpoints/drspaam_loc_best.pth" ]; then
        echo "=========================================================="
        echo "[STEP 3] Testing DR-SPAAM de lay AP..."
        echo "=========================================================="
        python -u test_temporal.py
    else
        echo "[ERROR] Train Loc that bai (Khong thay weight). Dung pipeline."
        exit 1
    fi
else
    echo "[ERROR] Train Seg that bai (Khong thay weight). Dung pipeline."
    exit 1
fi

echo "[SUCCESS] Toan bo DR-SPAAM pipeline hoan thanh luc: $(date)"