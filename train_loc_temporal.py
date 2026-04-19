import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from data.dataloader_temporal import FrogDataLoader, TemporalLocDataset, temporal_collate
from models.loc_model_temporal import TemporalLocModel1D
from utils import loc_model_loss_torch

# --- CẤU HÌNH HỆ THỐNG ---
PROJECT_ROOT   = "/home/s2410433/frog_project"
TRAINVAL_PATH  = os.path.join(PROJECT_ROOT, "data/frog_11-36_12-43_train_val.h5")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
PRETRAINED_SEG = os.path.join(CHECKPOINT_DIR, "drspaam_seg_best.pth")

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS     = 100
BATCH_SIZE = 64
T_WINDOW   = 5

def train():
    loader = FrogDataLoader(TRAINVAL_PATH, min_people=1)
    train_ds = TemporalLocDataset(loader, split=0, T=T_WINDOW, overlap_threshold=0.35)
    val_ds   = TemporalLocDataset(loader, split=1, T=T_WINDOW, overlap_threshold=0.35)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=temporal_collate, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=temporal_collate, num_workers=4)

    model = TemporalLocModel1D(num_anchors_per_sector=loader.NUM_ANCHORS_PER_SECTOR).to(DEVICE)

    # Load Backbone từ Segmentation
    if os.path.exists(PRETRAINED_SEG):
        print(f"Loading pretrained backbone from {PRETRAINED_SEG}")
        seg_state = torch.load(PRETRAINED_SEG, map_location=DEVICE)
        backbone_state = {k.replace("backbone.", ""): v for k, v in seg_state.items() if "backbone" in k}
        model.backbone.load_state_dict(backbone_state)
    else:
        print(f"[CẢNH BÁO] Không có {PRETRAINED_SEG}. Train từ đầu (scratch).")

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-5)

    best_val_loss = float("inf")
    print(f"Bắt đầu train DR-SPAAM Localization | Thiết bị: {DEVICE} | T={T_WINDOW}")

    for epoch in range(1, EPOCHS + 1):
        # --- Training ---
        model.train()
        total_train_loss = 0.0
        for scans_seq, gts in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Loc Train]"):
            scans_seq, gts = scans_seq.to(DEVICE), gts.to(DEVICE)
            optimizer.zero_grad()
            pred = model(scans_seq)
            loss = loc_model_loss_torch(gts, pred)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # --- Validation ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for scans_seq, gts in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Loc Val]"):
                scans_seq, gts = scans_seq.to(DEVICE), gts.to(DEVICE)
                pred = model(scans_seq)
                loss = loc_model_loss_torch(gts, pred)
                total_val_loss += loss.item()

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        avg_train = total_train_loss / len(train_loader)
        avg_val   = total_val_loss   / len(val_loader)

        print(f"Epoch {epoch}: Train = {avg_train:.4f} | Val = {avg_val:.4f} | LR = {current_lr:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_path = os.path.join(CHECKPOINT_DIR, "drspaam_loc_best.pth")
            torch.save(model.state_dict(), save_path)
            print(">>> Saved best DR-SPAAM localization model.")

if __name__ == "__main__":
    train()