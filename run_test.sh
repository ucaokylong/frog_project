#!/bin/bash
#PBS -q GPU-1
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N FROG_LFE_Full_Train

cd $PBS_O_WORKDIR

# 1. Kích hoạt môi trường
source $HOME/miniconda3/bin/activate research_venv

# 2. Kiểm tra và chỉ cài nếu thiếu (Smart Install)
echo "[*] Kiểm tra môi trường thư viện..."
REQUIRED_PKGS=("h5py" "numpy" "torch" "tqdm" "mlflow" "matplotlib" "scipy" "sklearn")

for pkg in "${REQUIRED_PKGS[@]}"; do
    # Thử import thư viện, nếu lỗi (exit code != 0) thì mới pip install
    python -c "import $pkg" &> /dev/null
    if [ $? -ne 0 ]; then
        echo "--> Thiếu $pkg, đang tiến hành cài đặt..."
        pip install $pkg
    else
        echo "--> $pkg: ĐÃ CÓ (Bỏ qua)"
    fi
done

# 3. Load CUDA
module load cuda/11.8

# 4. Kiểm tra GPU nhanh
nvidia-smi

# 5. Chạy Pipeline

echo "[STEP 3] Final Testing..."
python -u test.py

echo "[SUCCESS] Hoàn thành lúc: $(date)"