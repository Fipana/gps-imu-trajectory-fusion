"""
Custom loss functions
"""
import torch


def weighted_correction_loss(pred_correction, target_correction, mask, hdop, lambda_mag=1e-3):
    """
    HDOP-weighted loss for velocity corrections.
    
    Supervises only where GPS exists (mask=1) and weights by GPS quality.
    Also penalizes magnitude of corrections when HDOP is low (good GPS).
    
    Args:
        pred_correction: Predicted corrections (batch, seq_len, 2)
        target_correction: Target corrections (batch, seq_len, 2)
        mask: GPS availability mask (batch, seq_len, 1)
        hdop: HDOP values (batch, seq_len, 1)
        lambda_mag: Weight for magnitude regularization
        
    Returns:
        Scalar loss value
    """
    valid = mask > 0.5
    
    # Only compute loss where GPS exists
    target_safe = torch.where(valid.expand_as(target_correction),
                              target_correction,
                              pred_correction.detach())
    
    # Data fitting term: weighted by GPS quality
    sqerr = (pred_correction - target_safe) ** 2
    w_fit = 1.0 / (1.0 + hdop.clamp(0, 20.0))
    w_fit = torch.where(valid, w_fit, torch.zeros_like(w_fit))
    fit_term = (w_fit * sqerr.mean(dim=-1, keepdim=True)).sum() / (w_fit.sum() + 1e-8)
    
    # Magnitude regularizer: larger penalty when HDOP is small (good GPS)
    corr_mag2 = (pred_correction ** 2).sum(dim=-1, keepdim=True)
    w_reg = (1.0 / (1.0 + hdop.clamp(0, 20.0))).detach()
    reg_term = (w_reg * corr_mag2).mean()
    
    return fit_term + lambda_mag * reg_term
