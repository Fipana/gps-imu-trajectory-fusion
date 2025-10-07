# scripts/train.py
"""
Training script for velocity correction model (CLI-first; no env vars).

"""
import sys
import argparse
import yaml
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

# Local imports
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
    REPO = Path(__file__).resolve().parents[1]

    # Load YAML
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve paths (CLI > project_dir default > YAML repo-relative)
    # Seen data dir for training
    data_dir_seen = Path(args.data_dir) if args.data_dir else (REPO / cfg["data"]["seen_dir"])

    # RoNIN predictions for seen split
    if args.ronin_dir_seen:
        ronin_seen_dir = Path(args.ronin_dir_seen)
    elif args.project_dir:
        ronin_seen_dir = Path(args.project_dir) / "ronin_predictions" / "seen"
    else:
        ronin_seen_dir = REPO / cfg["data"]["ronin_seen_dir"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Seen data dir      :", data_dir_seen)
    print("RoNIN seen dir     :", ronin_seen_dir)

    # Load sequences (SEEN only for training)
    seq_names = [p.stem.replace("_gsn","") for p in Path(ronin_seen_dir).glob("*_gsn.npy")]
    sequences = load_aligned_sequences(seq_names, ronin_seen_dir, data_dir_seen)

    # Split
    train_seqs, val_seqs = create_stratified_split(sequences, cfg["training"]["train_ratio"])

    # Datasets / loaders
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

    # Model
    model = VelocityCorrectionLSTM(**cfg['model'])

    # Train
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
    parser.add_argument("--output", type=str, default="models/velocity_correction_model.pth")

    # Paths (CLI-first)
    parser.add_argument("--project_dir", type=str, help="If provided, uses {project_dir}/ronin_predictions/seen by default")
    parser.add_argument("--data_dir", type=str, help="SEEN dataset directory (e.g., .../seen_subjects_test_set)")
    parser.add_argument("--ronin_dir_seen", type=str, help="RoNIN outputs for SEEN (folder with *_gsn.npy)")

    args = parser.parse_args()
    main(args)
