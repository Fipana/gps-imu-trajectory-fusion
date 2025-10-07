"""
Evaluate a trained VC-LSTM on a split (seen/unseen)
"""
import sys
import argparse
from pathlib import Path
import torch
import os


sys.path.insert(0, str(Path(__file__).parent.parent))
from src.paths import get_paths

from src.models.velocity_lstm import VelocityCorrectionLSTM
from src.data.loader import load_aligned_sequences
from src.evaluation.metrics import evaluate_on_sequences

  
def main(args):
    if args.data_dir:    os.environ["DATA_DIR"]    = str(Path(args.data_dir))
    if args.model_dir:   os.environ["MODEL_DIR"]   = str(Path(args.model_dir))
    if args.project_dir: os.environ["PROJECT_DIR"] = str(Path(args.project_dir))

    P = get_paths()
    REPO = Path(__file__).resolve().parents[1]

    model_path = Path(args.model)
    ronin_dir  = Path(args.ronin_dir) if args.ronin_dir else (REPO / "ronin_predictions" / args.split)
    data_dir   = Path(args.data_dir)  if args.data_dir  else (REPO / "data" / f"{args.split}_subjects_test_set")
 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = VelocityCorrectionLSTM()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device).eval()

  
    seq_names = [p.stem.replace("_gsn", "") for p in Path(ronin_dir).glob("*_gsn.npy")]
    sequences = load_aligned_sequences(seq_names, ronin_dir, data_dir)

    df = evaluate_on_sequences(model, sequences, device, args.split.upper())
    out = Path(args.output_dir) / f"results_{args.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--ronin_dir", type=str)  # optional; else infer from repo
    parser.add_argument("--data_dir", type=str)   # optional; else infer from repo
    parser.add_argument("--split", type=str, default="unseen")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--project_dir", type=str)
    args = parser.parse_args()
    main(args)
