import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# Import từ các file local của bạn
from data import FrogDataLoader, SegDataset, temporal_collate
from models import SegModel1D
from utils import seg_mixed_loss_torch

# --- CẤU HÌNH HỆ THỐNG ---
PROJECT_ROOT = "/home/s2410433/frog_project"
TRAINVAL_PATH = os.path.join(PROJECT_ROOT, "data/frog_11-36_12-43_train_val.h5")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100       # Segmentation thường hội tụ nhanh hơn Loc
BATCH_SIZE = 64
LR = 1e-3
T = 5             # Độ dài chuỗi thời gian (khớp với Mamba sau này)

def train():
    print(f"--- BẮT ĐẦU TRAIN SEGMENTATION (T={T}) ---")
    loader = FrogDataLoader(TRAINVAL_PATH, min_people=0)
    
    # Dataset giờ nhận thêm tham số sequence_length
    train_ds = SegDataset(loader, split=0, sequence_length=T)
    val_ds = SegDataset(loader, split=1, sequence_length=T)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=temporal_collate, # Dùng collate chung cho chuỗi T
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=temporal_collate,
        num_workers=4,
        pin_memory=True
    )

    model = SegModel1D(glob=True).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Seg Train]")
        for scans, gts in pbar:
            scans, gts = scans.to(DEVICE), gts.to(DEVICE)

            optimizer.zero_grad()
            logits = model(scans) # Model xử lý (B, T, 1, L)
            loss = seg_mixed_loss_torch(gts, logits)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for scans, gts in val_loader:
                scans, gts = scans.to(DEVICE), gts.to(DEVICE)
                logits = model(scans)
                loss = seg_mixed_loss_torch(gts, logits)
                total_val_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader)

        print(f"Epoch {epoch}: Train Loss={avg_train:.4f}, Val Loss={avg_val:.4f}")

        # Lưu checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_path = os.path.join(CHECKPOINT_DIR, "lfe_seg_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f">>> Saved best segmentation model to {save_path}")

if __name__ == "__main__":
    train()