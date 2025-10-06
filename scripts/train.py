"""
Training script for velocity correction model
"""
import sys
import argparse
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.velocity_lstm import VelocityCorrectionLSTM
from src.data.dataset import VelocityCorrectionDataset
from src.data.loader import load_aligned_sequences
from src.training.trainer import train_correction_model


def create_stratified_split(sequences, train_ratio=0.8):
    """Split sequences by GPS coverage"""
    import numpy as np
    
    seq_with_coverage = []
    for seq in sequences:
        gps_mask = np.isfinite(seq['gps_pos']).all(axis=1)
        coverage = gps_mask.mean()
        seq_with_coverage.append((seq, coverage))
    
    seq_with_coverage.sort(key=lambda x: x[1])
    
    n = len(seq_with_coverage)
    low_cov = seq_with_coverage[:n//3]
    mid_cov = seq_with_coverage[n//3:2*n//3]
    high_cov = seq_with_coverage[2*n//3:]
    
    train_seqs, val_seqs = [], []
    for bin_seqs in [low_cov, mid_cov, high_cov]:
        n_train = int(len(bin_seqs) * train_ratio)
        train_seqs.extend([s[0] for s in bin_seqs[:n_train]])
        val_seqs.extend([s[0] for s in bin_seqs[n_train:]])
    
    print(f"Split: {len(train_seqs)} train, {len(val_seqs)} val")
    return train_seqs, val_seqs


def main(args):
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load sequences
    seen_names = [f.stem.replace('_gsn', '') 
                  for f in Path(config['data']['ronin_seen_dir']).glob('*_gsn.npy')]
    sequences = load_aligned_sequences(
        seen_names,
        config['data']['ronin_seen_dir'],
        config['data']['seen_dir']
    )
    
    # Split
    train_seqs, val_seqs = create_stratified_split(
        sequences,
        config['training']['train_ratio']
    )
    
    # Create datasets
    train_ds = VelocityCorrectionDataset(
        train_seqs,
        window=config['training']['window_size'],
        stride=config['training']['stride'],
        augment_gps_dropout=True
    )
    val_ds = VelocityCorrectionDataset(
        val_seqs,
        window=config['training']['window_size'],
        stride=config['training']['window_size'],
        augment_gps_dropout=False
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    model = VelocityCorrectionLSTM(**config['model'])
    
    # Train
    history = train_correction_model(
        train_loader,
        val_loader,
        model,
        device=device,
        num_epochs=config['training']['num_epochs'],
        lr=config['training']['learning_rate'],
        save_path=args.output
    )
    
    print(f"\nTraining complete! Model saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    parser.add_argument('--output', type=str, default='velocity_correction_model.pth')
    args = parser.parse_args()
    main(args)
