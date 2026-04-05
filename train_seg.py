import os
import torch
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from data import FrogDataLoader, SegDataset, seg_collate
from models import SegModel1D
from utils import seg_mixed_loss_torch

# --- CẤU HÌNH HỆ THỐNG ---
PROJECT_ROOT = "/home/s2410433/frog_project"
TRAINVAL_PATH = os.path.join(PROJECT_ROOT, "data/frog_11-36_12-43_train_val.h5")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3

# --- CẤU HÌNH MLFLOW (FIX LỖI) ---
# Sử dụng SQLite để tránh lỗi mất file meta.yaml trên hệ thống file Lustre
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
mlflow.set_experiment("FROG_Segmentation")

def train():
    loader = FrogDataLoader(TRAINVAL_PATH, min_people=0)
    train_ds = SegDataset(loader, split=0)
    val_ds = SegDataset(loader, split=1)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=seg_collate, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=seg_collate, num_workers=4)

    model = SegModel1D(glob=True).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    
    best_val_loss = float('inf')

    with mlflow.start_run(run_name="LFE_Seg_Baseline"):
        mlflow.log_params({
            "model": "SegModel1D",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "optimizer": "AdamW"
        })

        for epoch in range(1, EPOCHS + 1):
            # --- Training Phase ---
            model.train()
            total_train_loss = 0
            for scans, gts in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Seg Train]"):
                scans, gts = scans.to(DEVICE), gts.to(DEVICE)
                optimizer.zero_grad()
                logits = model(scans)
                loss = seg_mixed_loss_torch(gts, logits)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            # --- Validation Phase ---
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for scans, gts in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Seg Val]"):
                    scans, gts = scans.to(DEVICE), gts.to(DEVICE)
                    logits = model(scans)
                    loss = seg_mixed_loss_torch(gts, logits)
                    total_val_loss += loss.item()

            avg_train = total_train_loss / len(train_loader)
            avg_val = total_val_loss / len(val_loader)
            
            mlflow.log_metric("train_loss", avg_train, step=epoch)
            mlflow.log_metric("val_loss", avg_val, step=epoch)

            print(f"Epoch {epoch}: Train Loss={avg_train:.4f}, Val Loss={avg_val:.4f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                save_path = os.path.join(CHECKPOINT_DIR, "lfe_seg_best.pth")
                torch.save(model.state_dict(), save_path)
                mlflow.log_artifact(save_path)
                print(">>> Saved best segmentation model.")

if __name__ == "__main__":
    train()