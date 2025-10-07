# scripts/evaluate.py
"""
Evaluate a trained VC-LSTM using a YAML config (same as training).
"""
import sys
import argparse
from pathlib import Path
import torch
import os
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.paths import get_paths
from src.models.velocity_lstm import VelocityCorrectionLSTM
from src.data.loader import load_aligned_sequences
from src.evaluation.metrics import evaluate_on_sequences

def main(args):
    # Allow CLI to override env paths (optional)
    if args.data_dir:    os.environ["DATA_DIR"]    = str(Path(args.data_dir))
    if args.model_dir:   os.environ["MODEL_DIR"]   = str(Path(args.model_dir))
    if args.project_dir: os.environ["PROJECT_DIR"] = str(Path(args.project_dir))

    P = get_paths()
    REPO = Path(__file__).resolve().parents[1]

    # Load YAML (reuse train_config.yaml by default)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve repo-relative paths
    ronin_default = {
        "seen":   REPO / cfg["data"]["ronin_seen_dir"],
        "unseen": REPO / cfg["data"]["ronin_unseen_dir"],
    }
    data_default = {
        "seen":   REPO / cfg["data"]["seen_dir"],
        "unseen": REPO / cfg["data"]["unseen_dir"],
    }

    ronin_dir = Path(args.ronin_dir) if args.ronin_dir else ronin_default[args.split]
    data_dir  = Path(args.data_dir)  if args.data_dir  else data_default[args.split]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    print("Resolved paths:", {k: str(v) for k, v in P.items()})
    print("Ronin dir:", ronin_dir)
    print("Data dir :", data_dir)

    # Model path: CLI overrides YAML; else default to models/velocity_correction_model.pth
    model_path = Path(args.model) if args.model \
        else (REPO / "models" / "velocity_correction_model.pth")

    model = VelocityCorrectionLSTM(**cfg.get("model", {}))
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()

    # Load sequences
    seq_names = [p.stem.replace("_gsn", "") for p in Path(ronin_dir).glob("*_gsn.npy")]
    sequences = load_aligned_sequences(seq_names, ronin_dir, data_dir)

    # Fusion params from YAML
    fusion_cfg = cfg.get("fusion", {})
    hdop_threshold = float(fusion_cfg.get("hdop_threshold", 2.0))
    max_correction = float(fusion_cfg.get("max_correction", 0.15))

    df = evaluate_on_sequences(
        model, sequences, device, args.split.upper(),
        hdop_threshold=hdop_threshold,
        max_correction=max_correction,
    )

    out = Path(args.output_dir) / f"results_{args.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nResults saved to {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--model", type=str)       # optional; else models/velocity_correction_model.pth
    parser.add_argument("--ronin_dir", type=str)   # optional; else from YAML
    parser.add_argument("--data_dir", type=str)    # optional; else from YAML
    parser.add_argument("--split", type=str, choices=["seen","unseen"], default="unseen")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--model_dir", type=str)   # optional env override
    parser.add_argument("--project_dir", type=str) # optional env override
    args = parser.parse_args()
    main(args)

