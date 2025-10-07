# scripts/evaluate.py
"""
Evaluate a trained VC-LSTM using a YAML config (CLI-first; no env vars).

"""
import sys
import argparse
from pathlib import Path
import torch
import yaml

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.velocity_lstm import VelocityCorrectionLSTM
from src.data.loader import load_aligned_sequences
from src.evaluation.metrics import evaluate_on_sequences


def main(args):
    REPO = Path(__file__).resolve().parents[1]

    # Load YAML
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Choose split
    split = args.split.lower()
    assert split in ("seen", "unseen")

    # Resolve dataset dir for this split
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        key = "seen_dir" if split == "seen" else "unseen_dir"
        data_dir = REPO / cfg["data"][key]

    # Resolve RoNIN dir for this split
    if split == "seen":
        if args.ronin_dir_seen:
            ronin_dir = Path(args.ronin_dir_seen)
        elif args.project_dir:
            ronin_dir = Path(args.project_dir) / "ronin_predictions" / "seen"
        else:
            ronin_dir = REPO / cfg["data"]["ronin_seen_dir"]
    else:
        if args.ronin_dir_unseen:
            ronin_dir = Path(args.ronin_dir_unseen)
        elif args.project_dir:
            ronin_dir = Path(args.project_dir) / "ronin_predictions" / "unseen"
        else:
            ronin_dir = REPO / cfg["data"]["ronin_unseen_dir"]

    # Model path
    model_path = Path(args.model) if args.model else (REPO / "models" / "velocity_correction_model.pth")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Split        :", split)
    print("Data dir     :", data_dir)
    print("RoNIN dir    :", ronin_dir)
    print("Model        :", model_path)

    # Load model
    model = VelocityCorrectionLSTM(**cfg.get("model", {}))
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()

    # Load sequences
    seq_names = [p.stem.replace("_gsn", "") for p in Path(ronin_dir).glob("*_gsn.npy")]
    sequences = load_aligned_sequences(seq_names, ronin_dir, data_dir)

    # Fusion params
    fusion_cfg = cfg.get("fusion", {})
    df = evaluate_on_sequences(
        model, sequences, device, split.upper(),
        hdop_threshold=float(fusion_cfg.get("hdop_threshold", 2.0)),
        max_correction=float(fusion_cfg.get("max_correction", 0.15)),
    )

    out = Path(args.output_dir) / f"results_{split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--model", type=str, help="Path to trained .pth")
    parser.add_argument("--split", type=str, choices=["seen","unseen"], default="unseen")
    parser.add_argument("--output_dir", type=str, default="results")

    # Paths (CLI-first)
    parser.add_argument("--project_dir", type=str, help="If provided, uses {project_dir}/ronin_predictions/{split}")
    parser.add_argument("--data_dir", type=str, help="Dataset dir for the chosen split")
    parser.add_argument("--ronin_dir_seen", type=str, help="Override RoNIN dir for SEEN split")
    parser.add_argument("--ronin_dir_unseen", type=str, help="Override RoNIN dir for UNSEEN split")

    args = parser.parse_args()
    main(args)
