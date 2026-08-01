import os
import torch
# KHAI BÁO CHUẨN TỪ torch.cuda.amp - AN TOÀN TRÊN MỌI PHIÊN BẢN PYTORCH
from torch.cuda.amp import autocast, GradScaler  
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from data.dataloader_temporal import FrogDataLoader, TemporalLocDataset, temporal_collate
from models.loc_model_temporal import TemporalLocModel1D
from utils.loss_temporal import loc_model_loss_torch

# --- CẤU HÌNH HỆ THỐNG ---
PROJECT_ROOT   = "/home/s2410433/frog_project/temporal_model"
TRAINVAL_PATH  = os.path.join(PROJECT_ROOT, "data/frog_11-36_12-43_train_val.h5")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS     = 100
BATCH_SIZE = 32 
T_WINDOW   = 5
EARLY_STOP_PATIENCE = 10 

def train():
    loader = FrogDataLoader(TRAINVAL_PATH, min_people=1)
    train_ds = TemporalLocDataset(loader, split=0, T=T_WINDOW)
    val_ds   = TemporalLocDataset(loader, split=1, T=T_WINDOW)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=temporal_collate, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=temporal_collate, num_workers=4)

    model = TemporalLocModel1D().to(DEVICE)
    print("[INFO] Khởi tạo mô hình mới hoàn toàn. Train từ đầu (Training from scratch).")

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-5)
    
    # KHỞI TẠO SCALER CƠ BẢN
    scaler = GradScaler()

    best_val_loss = float("inf")
    epochs_no_improve = 0 
    save_path = os.path.join(CHECKPOINT_DIR, "drspaam_loc_cutout_best.pth")

    print(f"Bắt đầu train LOCALIZATION (CUTOUT) | Thiết bị: {DEVICE} | T={T_WINDOW}")
    print(f"File trọng số sẽ được lưu đè tại: {save_path}")

    for epoch in range(1, EPOCHS + 1):
        # --- Training ---Q
        model.train()
        total_train_loss = 0.0
        for scans_seq, gts in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Loc Train]"):
            scans_seq, gts = scans_seq.to(DEVICE), gts.to(DEVICE)
            optimizer.zero_grad()
            
            # GỌI AUTOCAST CHUẨN
            with autocast():
                pred = model(scans_seq)
                loss = loc_model_loss_torch(gts, pred)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()

        # --- Validation ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for scans_seq, gts in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Loc Val]"):
                scans_seq, gts = scans_seq.to(DEVICE), gts.to(DEVICE)
                
                with autocast():
                    pred = model(scans_seq)
                    loss = loc_model_loss_torch(gts, pred)
                    
                total_val_loss += loss.item()

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        avg_train = total_train_loss / len(train_loader)
        avg_val   = total_val_loss   / len(val_loader)

        print(f"Epoch {epoch}: Train = {avg_train:.4f} | Val = {avg_val:.4f} | LR = {current_lr:.6f}")

        # --- KIỂM TRA EARLY STOPPING VÀ LƯU MODEL ---
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0 
            torch.save(model.state_dict(), save_path)
            print(f">>> Đã lưu model tốt nhất mới (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"[Early Stopping] Chưa có cải thiện: {epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs liên tiếp.")
            
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"\n!!! KÍCH HOẠT EARLY STOPPING SAU {epoch} EPOCHS !!!")
                print(f"Loss đã không giảm trong {EARLY_STOP_PATIENCE} epoch liên tiếp. Dừng huấn luyện.")
                break

if __name__ == "__main__":
    train()