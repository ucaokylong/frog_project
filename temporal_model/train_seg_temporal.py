import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from data.dataloader_temporal import FrogDataLoader, TemporalSegDataset, temporal_collate
from models.seg_model_temporal import TemporalSegModel1D
from utils.loss_temporal import seg_mixed_loss_torch

# --- CẤU HÌNH HỆ THỐNG ---
PROJECT_ROOT = "/home/s2410433/frog_project"
TRAINVAL_PATH = os.path.join(PROJECT_ROOT, "data/frog_11-36_12-43_train_val.h5")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
T_WINDOW = 5

def train():
    loader = FrogDataLoader(TRAINVAL_PATH, min_people=0)
    train_ds = TemporalSegDataset(loader, split=0, T=T_WINDOW)
    val_ds = TemporalSegDataset(loader, split=1, T=T_WINDOW)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=temporal_collate, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=temporal_collate, num_workers=4)

    model = TemporalSegModel1D().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

    best_val_loss = float("inf")
    print(f"Bắt đầu train SEGMENTATION (CUTOUT) | Thiết bị: {DEVICE} | T={T_WINDOW}")

    for epoch in range(1, EPOCHS + 1):
        # --- Training ---
        model.train()
        total_train_loss = 0.0
        for scans_seq, gts in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Seg Train]"):
            scans_seq, gts = scans_seq.to(DEVICE), gts.to(DEVICE)
            optimizer.zero_grad()
            logits = model(scans_seq)
            loss = seg_mixed_loss_torch(gts, logits)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # --- Validation ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for scans_seq, gts in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Seg Val]"):
                scans_seq, gts = scans_seq.to(DEVICE), gts.to(DEVICE)
                logits = model(scans_seq)
                loss = seg_mixed_loss_torch(gts, logits)
                total_val_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader)

        print(f"Epoch {epoch}: Train Loss = {avg_train:.4f} | Val Loss = {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            # TÊN FILE ĐÃ ĐƯỢC ĐỔI ĐỂ PHÂN BIỆT VỚI MODEL CŨ
            save_path = os.path.join(CHECKPOINT_DIR, "drspaam_seg_cutout_best.pth")
            torch.save(model.state_dict(), save_path)
            print(">>> Saved best Segmentation (Cutout) model.")

if __name__ == "__main__":
    train()