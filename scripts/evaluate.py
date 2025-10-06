"""
Evaluation script
"""
import sys
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.velocity_lstm import VelocityCorrectionLSTM
from src.data.loader import load_aligned_sequences
from src.evaluation.metrics import evaluate_on_sequences


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = VelocityCorrectionLSTM()
    model.load_state_dict(torch.load(args.model))
    model = model.to(device)
    model.eval()

    # Load sequences
    seq_names = [f.stem.replace('_gsn', '')
                 for f in Path(args.ronin_dir).glob('*_gsn.npy')]
    sequences = load_aligned_sequences(seq_names, args.ronin_dir, args.data_dir)

    # Evaluate
    results = evaluate_on_sequences(model, sequences, device, args.split.upper())

    # Save results
    output_path = Path(args.output_dir) / f'results_{args.split}.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--ronin_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='unseen')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    main(args)
