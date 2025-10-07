"""Training loop"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .loss import weighted_correction_loss

def train_correction_model(train_loader, val_loader, model, device='cuda',
                           num_epochs=50, lr=1e-3, weight_decay=1e-4, 
                           patience=8, save_path='models/velocity_correction_model.pth'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            X = batch['X'].to(device)
            Y = batch['Y'].to(device)
            M = batch['M'].to(device)
            hdop = X[:, :, 6:7]

            optimizer.zero_grad()
            pred = model(X)
            loss = weighted_correction_loss(pred, Y, M, hdop)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train = float(np.mean(train_losses))
        history['train'].append(avg_train)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                X = batch['X'].to(device)
                Y = batch['Y'].to(device)
                M = batch['M'].to(device)
                hdop = X[:, :, 6:7]
                pred = model(X)
                loss = weighted_correction_loss(pred, Y, M, hdop)
                val_losses.append(loss.item())

        avg_val = float(np.mean(val_losses)) if val_losses else float('inf')
        history['val'].append(avg_val)

        print(f"Epoch {epoch+1}: Train={avg_train:.4f}, Val={avg_val:.4f}")
        scheduler.step(avg_val)

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
