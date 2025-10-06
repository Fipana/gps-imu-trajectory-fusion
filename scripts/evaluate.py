"""
Evaluate a trained VC-LSTM on a split (seen/unseen)
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
    print(f"Using device: {device}")

    model = VelocityCorrectionLSTM()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device).eval()

    seq_names = [p.stem.replace("_gsn", "") for p in Path(args.ronin_dir).glob("*_gsn.npy")]
    sequences = load_aligned_sequences(seq_names, args.ronin_dir, args.data_dir)

    df = evaluate_on_sequences(model, sequences, device, args.split.upper())
    out = Path(args.output_dir) / f"results_{args.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=str)
    ap.add_argument("--ronin_dir", required=True, type=str)
    ap.add_argument("--data_dir", required=True, type=str)
    ap.add_argument("--split", type=str, default="unseen")
    ap.add_argument("--output_dir", type=str, default="results")
    main(ap.parse_args())
