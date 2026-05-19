import torch
import torch.nn.functional as F

def dice_loss_torch(y_true, y_pred, smooth=1.0):
    num_TP       = (y_true * y_pred).sum(dim=(1, 2))
    num_pos_gt   = y_true.sum(dim=(1, 2))
    num_pos_pred = y_pred.sum(dim=(1, 2))
    dice = 1.0 - (2.0 * num_TP + smooth) / (num_pos_gt + num_pos_pred + smooth)
    return dice.mean()

def weighted_ce_loss_torch(y_true, logits, beta=1.5):
    pos_weight = torch.tensor([beta], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, y_true, pos_weight=pos_weight)
    return bce

def seg_mixed_loss_torch(y_true, logits):
    probs = torch.sigmoid(logits)
    dice = dice_loss_torch(y_true, probs)
    ce   = weighted_ce_loss_torch(y_true, logits)
    return 0.5 * (dice + ce)

def loc_model_loss_torch(y_true, y_pred):
    """
    Hàm loss cho Localization Cutout.
    y_true/y_pred: (B, 720, 3) -> [objectness, dx, dy]
    """
    gt_obj = y_true[..., 0]
    gt_reg = y_true[..., 1:3]
    pr_obj = y_pred[..., 0]
    pr_reg = y_pred[..., 1:3]

    # Mask cho các cutout có người (1.0) và không có người (0.0), bỏ qua (-1.0)
    pos_mask = (gt_obj > 0.5).float()
    valid_mask = (gt_obj >= 0.0).float()
    num_pos = pos_mask.sum() + 1e-6

    # 1. Classification Loss (BCE)
    cls_loss_all = F.binary_cross_entropy_with_logits(pr_obj, pos_mask, reduction='none')
    cls_loss = (valid_mask * cls_loss_all).sum() / (valid_mask.sum() + 1e-6)

    # 2. Regression Loss (Smooth L1)
    sigma_sq = 9.0 
    diff = torch.abs(gt_reg - pr_reg)
    threshold = 1.0 / sigma_sq
    
    piecewise = (diff < threshold).float()
    reg_loss_quad = 0.5 * diff**2 * sigma_sq
    reg_loss_lin = diff - 0.5 / sigma_sq
    
    loss_all = piecewise * reg_loss_quad + (1.0 - piecewise) * reg_loss_lin
    # Chỉ tính Regression loss trên các cutout CÓ người (pos_mask)
    reg_loss = (pos_mask.unsqueeze(-1) * loss_all).sum() / num_pos

    return cls_loss + reg_loss