"""
Training loop for velocity correction model
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .loss import weighted_correction_loss


def train_correction_model(train_loader, val_loader, model, device='cuda', 
                          num_epochs=50, lr=1e-3, save_path='model.pth'):
    """
    Train the velocity correction model.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        model: Model to train
        device: Device to train on
        num_epochs: Number of epochs
        lr: Learning rate
        save_path: Path to save best model
        
    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    best_val_loss = float('inf')
    patience, patience_counter = 8, 0
    history = {'train': [], 'val': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            X = batch['X'].to(device)
            Y = batch['Y'].to(device)
            M = batch['M'].to(device)
            hdop = X[:, :, 6:7]
            
            optimizer.zero_grad()
            pred_correction = model(X)
            loss = weighted_correction_loss(pred_correction, Y, M, hdop)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train = np.mean(train_losses)
        history['train'].append(avg_train)
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                X = batch['X'].to(device)
                Y = batch['Y'].to(device)
                M = batch['M'].to(device)
                hdop = X[:, :, 6:7]
                
                pred_correction = model(X)
                loss = weighted_correction_loss(pred_correction, Y, M, hdop)
                val_losses.append(loss.item())
        
        avg_val = np.mean(val_losses) if val_losses else float('inf')
        history['val'].append(avg_val)
        
        print(f"Epoch {epoch+1}: Train={avg_train:.4f}, Val={avg_val:.4f}")
        
        scheduler.step(avg_val)
        
        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  → Best model saved (val={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return history
