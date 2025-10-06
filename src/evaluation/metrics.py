"""Evaluation metrics and loop"""
import numpy as np
from tqdm import tqdm


def compute_rmse(pred, gt):
    pred = np.asarray(pred, float)
    gt = np.asarray(gt, float)
    m = np.isfinite(pred).all(axis=1) & np.isfinite(gt).all(axis=1)
    if m.sum() == 0: return np.nan
    d = pred[m] - gt[m]
    return float(np.sqrt((d**2).sum(axis=1).mean()))


def compute_rte(pred, gt, time_s, window_sec=60.0):
    t = np.asarray(time_s, float)
    pred = np.asarray(pred, float)
    gt = np.asarray(gt, float)
    valid = np.isfinite(pred).all(axis=1) & np.isfinite(gt).all(axis=1)
    if valid.sum() < 10: return np.nan

    dt = np.diff(t, prepend=t[0])
    dt[dt <= 0] = np.median(dt[dt > 0]) if (dt > 0).any() else 0.1
    step = max(2, int(np.round(window_sec / np.nanmedian(dt[np.isfinite(dt)]))))
    errs = []
    for i in range(0, len(t) - step, step // 2 if step > 2 else 1):
        sl = slice(i, i+step)
        m = valid[sl]
        if m.sum() < max(5, step // 3): continue
        d = pred[sl][m] - gt[sl][m]
        errs.append(np.sqrt((d**2).sum(axis=1).mean()))
    return float(np.mean(errs)) if errs else np.nan


def evaluate_on_sequences(model, sequences, device, split_name="TEST"):
    import pandas as pd
    from ..inference.predictor import predict_sequence_with_corrections

    rows = []
    print(f"Evaluating on {split_name} ({len(sequences)} sequences)")
    print("-"*50)
    for seq in tqdm(sequences, desc=f"Evaluating {split_name}"):
        fused_pos, _, gps_mask = predict_sequence_with_corrections(model, seq, device)
        rmse_fused = compute_rmse(fused_pos, seq['gt_pos'])
        rte_fused  = compute_rte(fused_pos, seq['gt_pos'], seq['time'])
        rmse_ronin = compute_rmse(seq['ronin_pos'], seq['gt_pos'])
        rte_ronin  = compute_rte(seq['ronin_pos'], seq['gt_pos'], seq['time'])
        gps_cov = float(gps_mask.mean() * 100.0)
        rows.append({
            "seq_name": seq['seq_name'],
            "rmse_fused": rmse_fused,
            "rte_fused": rte_fused,
            "rmse_ronin": rmse_ronin,
            "rte_ronin": rte_ronin,
            "gps_coverage": gps_cov,
            "improvement_rmse": ((rmse_ronin - rmse_fused)/rmse_ronin*100) if rmse_ronin>0 else 0.0,
        })
    df = pd.DataFrame(rows)
    print(f"\n{split_name} RESULTS")
    print("-"*50)
    print(f"Fused RMSE: {df['rmse_fused'].mean():.3f} ± {df['rmse_fused'].std():.3f} m")
    print(f"Fused RTE:  {df['rte_fused'].mean():.3f} ± {df['rte_fused'].std():.3f} m")
    print(f"RoNIN RMSE: {df['rmse_ronin'].mean():.3f} ± {df['rmse_ronin'].std():.3f} m")
    print(f"Improvement: {df['improvement_rmse'].mean():.1f}%")
    return df
