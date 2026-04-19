#!/bin/bash
#SBATCH -p GPU-1                        # GPU partition (A40 nodes)
#SBATCH -J FROG_LFE_Full_Train          # Job name
#SBATCH -n 4                            # Number of CPU cores
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --mem=32G                       # Memory
#SBATCH --time=72:00:00                 # Walltime (max 1 week for GPU-1)
#SBATCH -o FROG_LFE_Full_Train-%j.log   # Stdout + Stderr merged (replaces -j oe)

cd $SLURM_SUBMIT_DIR                    # Replaces $PBS_O_WORKDIR

# 1. Activate environment
source $HOME/miniconda3/bin/activate research_venv

# 2. Smart install check
echo "[*] Kiem tra moi truong thu vien..."
REQUIRED_PKGS=("h5py" "numpy" "torch" "tqdm" "mlflow" "matplotlib" "scipy" "sklearn")

for pkg in "${REQUIRED_PKGS[@]}"; do
    python -c "import $pkg" &> /dev/null
    if [ $? -ne 0 ]; then
        echo "--> Thieu $pkg, dang tien hanh cai dat..."
        # sklearn duoc cai bang ten scikit-learn
        if [ "$pkg" == "sklearn" ]; then
            pip install scikit-learn
        else
            pip install $pkg
        fi
    else
        echo "--> $pkg: DA CO (Bo qua)"
    fi
done

# 3. Load CUDA
module load cuda/11.8

# 4. Kiem tra GPU
nvidia-smi

# 5. Chay Pipeline
echo "[STEP 1] Training Segmentation..."
python -u train_seg.py

echo "[STEP 2] Training Localization..."
python -u train_loc.py

echo "[STEP 3] Final Testing..."
python -u test.py

echo "[SUCCESS] Hoan thanh luc: $(date)"