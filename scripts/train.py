"""
Training script for velocity correction model
"""
import sys
import argparse
import yaml
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.velocity_lstm import VelocityCorrectionLSTM
from src.data.dataset import VelocityCorrectionDataset
from src.data.loader import load_aligned_sequences


def create_stratified_split(sequences, train_ratio=0.8):
    seq_cov = []
    for s in sequences:
        gps_mask = np.isfinite(s['gps_pos']).all(axis=1)
        seq_cov.append((s, float(gps_mask.mean())))
    seq_cov.sort(key=lambda x: x[1])
    n = len(seq_cov)
    low, mid, high = seq_cov[:n//3], seq_cov[n//3:2*n//3], seq_cov[2*n//3:]
    train, val = [], []
    for bucket in (low, mid, high):
        k = int(len(bucket) * train_ratio)
        train += [x[0] for x in bucket[:k]]
        val   += [x[0] for x in bucket[k:]]
    print(f"Split: {len(train)} train, {len(val)} val")
    return train, val


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    seen_dir = cfg['data']['seen_dir']
    ronin_seen_dir = cfg['data']['ronin_seen_dir']

    seq_names = [p.stem.replace("_gsn","") for p in Path(ronin_seen_dir).glob("*_gsn.npy")]
    sequences = load_aligned_sequences(seq_names, ronin_seen_dir, seen_dir)

    train_seqs, val_seqs = create_stratified_split(sequences, cfg['training']['train_ratio'])

    train_ds = VelocityCorrectionDataset(
        train_seqs,
        window=cfg['training']['window_size'],
        stride=cfg['training']['stride'],
        augment_gps_dropout=True
    )
    val_ds = VelocityCorrectionDataset(
        val_seqs,
        window=cfg['training']['window_size'],
        stride=cfg['training']['window_size'],
        augment_gps_dropout=False
    )

    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=0)

    model = VelocityCorrectionLSTM(**cfg['model'])

    from src.training.trainer import train_correction_model
    history = train_correction_model(
        train_loader, val_loader, model, device=device,
        num_epochs=cfg['training']['num_epochs'],
        lr=cfg['training']['learning_rate'],
        save_path=args.output
    )
    print(f"\nTraining complete. Best model saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--output", type=str, default="velocity_correction_model.pth")
    main(parser.parse_args())
