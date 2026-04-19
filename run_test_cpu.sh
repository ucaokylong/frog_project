#!/bin/bash
#PBS -q SINGLE
#PBS -l select=1:ncpus=16:mpiprocs=16
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N FROG_LFE_Full_Test_CPU

cd $PBS_O_WORKDIR

# 1. Kích hoạt môi trường
source $HOME/miniconda3/bin/activate research_venv

# 2. Kiểm tra và chỉ cài nếu thiếu (Smart Install)
echo "[*] Kiểm tra môi trường thư viện..."
REQUIRED_PKGS=("h5py" "numpy" "torch" "tqdm" "mlflow" "matplotlib" "scipy" "sklearn")

for pkg in "${REQUIRED_PKGS[@]}"; do
    python -c "import $pkg" &> /dev/null
    if [ $? -ne 0 ]; then
        echo "--> Thiếu $pkg, đang tiến hành cài đặt..."
        pip install $pkg
    else
        echo "--> $pkg: ĐÃ CÓ (Bỏ qua)"
    fi
done

# 3. Force CPU-only (vô hiệu hóa GPU)
export CUDA_VISIBLE_DEVICES=""

# 4. Chạy Pipeline
echo "[STEP] Final Testing (CPU only)..."
python -u test.py

echo "[SUCCESS] Hoàn thành lúc: $(date)"
