import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from data.dataloader_mamba import FrogDataLoader, LocDataset, temporal_collate
from models.loc_model_mamba import LocModel1D
from utils.loss_mamba import loc_model_loss_torch

# --- CẤU HÌNH HỆ THỐNG ---
PROJECT_ROOT   = "/home/s2410433/frog_project"
TRAINVAL_PATH  = os.path.join(PROJECT_ROOT, "data/frog_11-36_12-43_train_val.h5")
PRETRAINED_SEG = os.path.join(PROJECT_ROOT, "checkpoints/lfe_seg_best.pth")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS     = 100
BATCH_SIZE = 32 # Nếu bị Out of Memory (OOM) khi tăng T lên 15, hãy giảm xuống 16 hoặc 24 nhé
T          = 15 # Tăng chiều dài chuỗi để Mamba học quỹ đạo chuyển động rõ ràng hơn

def train():
    print(f"--- BẮT ĐẦU TRAIN LOCALIZATION + MAMBA (T={T}) END-TO-END ---")
    loader   = FrogDataLoader(TRAINVAL_PATH, min_people=1)
    train_ds = LocDataset(loader, split=0, sequence_length=T, overlap_threshold=0.35)
    val_ds   = LocDataset(loader, split=1, sequence_length=T, overlap_threshold=0.35)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=temporal_collate, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=temporal_collate, num_workers=4, pin_memory=True)

    model = LocModel1D(
        num_anchors_per_sector=loader.NUM_ANCHORS_PER_SECTOR,
        glob=True,
        d_model=96 
    ).to(DEVICE)

    # --- Load pretrained backbone từ SegModel ---
    if os.path.exists(PRETRAINED_SEG):
        print(f"Loading weights from {PRETRAINED_SEG}")
        seg_state = torch.load(PRETRAINED_SEG, map_location=DEVICE)
        
        backbone_state = {
            k.replace("backbone.", ""): v
            for k, v in seg_state.items() if "backbone" in k
        }
        # strict=False giúp load an toàn
        model.backbone.load_state_dict(backbone_state, strict=False)
        print(">>> Backbone weights loaded successfully. Ready for Fine-tuning.")
    else:
        print("!!! Warning: Pretrained Seg weights not found. Training from scratch.")

    # --- DIFFERENTIAL LEARNING RATES CỰC KỲ QUAN TRỌNG KHI KHÔNG FREEZE ---
    mamba_params = []
    base_params = []
    
    for name, param in model.named_parameters():
        if "temporal_processor" in name:
            mamba_params.append(param)
        else:
            base_params.append(param) # Gồm cả Backbone và Head

    # Backbone/Head chạy nhanh (1e-3), Mamba chạy chậm để từ từ thích nghi (1e-4)
    optimizer = AdamW([
        {'params': base_params, 'lr': 1e-3, 'weight_decay': 1e-3},
        {'params': mamba_params, 'lr': 1e-4, 'weight_decay': 5e-3} 
    ])
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Loc Train]")
        for scans, gts in pbar:
            scans, gts = scans.to(DEVICE), gts.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(scans)
            loss = loc_model_loss_torch(gts, pred)
            loss.backward()
            
            # Clip gradient bảo vệ mạng lúc đầu khi Mamba chưa ổn định
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for scans, gts in val_loader:
                scans, gts = scans.to(DEVICE), gts.to(DEVICE)
                pred = model(scans)
                loss = loc_model_loss_torch(gts, pred)
                total_val_loss += loss.item()

        scheduler.step()
        
        # Log ra LR của cả 2 phần để dễ theo dõi
        current_lr_base = optimizer.param_groups[0]['lr']
        current_lr_mamba = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else current_lr_base

        avg_train = total_train_loss / len(train_loader)
        avg_val   = total_val_loss   / len(val_loader)

        print(f"Epoch {epoch}: Train={avg_train:.4f} | Val={avg_val:.4f} | LR_Base={current_lr_base:.6f} | LR_Mamba={current_lr_mamba:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_path = os.path.join(CHECKPOINT_DIR, "lfe_mamba_loc_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f">>> Saved best Mamba-Loc model to {save_path}")

if __name__ == "__main__":
    train()