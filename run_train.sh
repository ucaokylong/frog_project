#!/bin/bash
#SBATCH -p GPU-1
#SBATCH -J FROG_LFE_PPN
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
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
# BƯỚC 3: PIPELINE LFE-PPN (SEG -> LOC -> TEST)
# ==========================================

echo "=========================================================="
echo "[STEP 1] Training LFE Segmentation (Basic)..."
echo "=========================================================="
python -u train_seg_basic.py

# Kiểm tra nếu train Seg thành công (có file lfe_seg_best.pth) thì mới chạy tiếp Loc
if [ -f "checkpoints/lfe_seg_best.pth" ]; then
    echo "=========================================================="
    echo "[STEP 2] Training LFE Localization PPN (Basic)..."
    echo "=========================================================="
    python -u train_loc_basic.py
    
    # Kiểm tra nếu train Loc thành công (có file lfe_ppn_best.pth) thì mới chạy Test
    if [ -f "checkpoints/lfe_ppn_best.pth" ]; then
        echo "=========================================================="
        echo "[STEP 3] Testing LFE-PPN Pipeline de lay mAP..."
        echo "=========================================================="
        python -u test_basic.py
    else
        echo "[ERROR] Train Loc PPN that bai (Khong thay weight lfe_ppn_best.pth). Dung pipeline."
        exit 1
    fi
else
    echo "[ERROR] Train Seg that bai (Khong thay weight lfe_seg_best.pth). Dung pipeline."
    exit 1
fi

echo "[SUCCESS] Toan bo LFE-PPN pipeline hoan thanh luc: $(date)"