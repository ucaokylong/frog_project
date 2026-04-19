#!/bin/bash
#SBATCH -p GPU-1
#SBATCH -J FROG_Mamba_Final
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH -o FROG_Final_Output-%j.log

cd $SLURM_SUBMIT_DIR

# 1. KÍCH HOẠT MÔI TRƯỜNG (Sử dụng cách thức an toàn nhất cho Slurm)
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate mamba_venv

# 2. THIẾT LẬP CUDA RUNTIME (Để Mamba tìm thấy nhân CUDA khi tính toán)
module load cuda/11.8
export CUDA_HOME=$CUDA_ROOT
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 3. KIỂM TRA NHANH (Để chắc chắn trong Log là mọi thứ vẫn ổn)
echo "[*] Thoi gian bat dau: $(date)"
echo "[*] Python: $(which python)"
python -c "import torch; import mamba_ssm; print('Check OK: Mamba da san sang de train!')"

# 4. CHẠY PIPELINE TRAIN & TEST
echo "=========================================================="
echo "[STEP 2] Training Localization with Mamba (T=15)..."
echo "=========================================================="
# -u để log đẩy ra file ngay lập tức, không bị nghẽn
python -u train_loc.py

# Nếu train thành công (tạo ra file weight), tự động chạy Test luôn
if [ -f "checkpoints/lfe_mamba_loc_best.pth" ]; then
    echo "=========================================================="
    echo "[STEP 3] Training xong! Dang chay Test de lay ket qua AP..."
    echo "=========================================================="
    python -u test.py
else
    echo "[ERROR] Khong tim thay weight sau khi train. Kiem tra lai train_loc.py"
    exit 1
fi

echo "[SUCCESS] Toan bo pipeline hoan thanh luc: $(date)"