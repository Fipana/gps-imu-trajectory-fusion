"""Data loading utilities"""
import numpy as np
import h5py
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from ..utils.helpers import safe_dt, ffill_2d

def load_sequence_data(seq_name, ronin_dir, data_root):
    """Load one sequence with RoNIN, GT, GPS data"""
    ronin_path = Path(ronin_dir)
    data_path = Path(data_root) / seq_name

    ronin_file = ronin_path / f"{seq_name}_gsn.npy"
    if not ronin_file.exists():
        return None

    ronin_pos = np.load(ronin_file).astype(np.float32)[:, :2]

    with h5py.File(data_path / "data.hdf5", "r") as f:
        gt_pos = np.array(f["pose"]["tango_pos"])[:, :2].astype(np.float32)
        time = np.array(f["synced"]["time"]).astype(np.float32)

    start_frame = 0
    info_file = data_path / "info.json"
    if info_file.exists():
        with open(info_file, "r") as f:
            start_frame = int(json.load(f).get("start_frame", 0))

    T = min(len(ronin_pos), len(gt_pos), len(time))
    ronin_pos = ronin_pos[:T]
    gt_pos = gt_pos[start_frame:start_frame+T]
    time = time[start_frame:start_frame+T]

    # Load GPS data
    gps_aligned = np.full((T, 3), np.nan)
    gps_file = data_path / "data_gps.csv"
    if gps_file.exists():
        gps_df = pd.read_csv(gps_file)
        rename = {}
        for col in gps_df.columns:
            low = col.lower()
            if 'time' in low: rename[col] = 'time_s'
            elif low.startswith('x') or 'x_m' in low: rename[col] = 'x_m'
            elif low.startswith('y') or 'y_m' in low: rename[col] = 'y_m'
            elif 'hdop' in low: rename[col] = 'HDOP'
        gps_df = gps_df.rename(columns=rename)
        if 'HDOP' not in gps_df.columns:
            gps_df['HDOP'] = 5.0

        if all(k in gps_df.columns for k in ['time_s', 'x_m', 'y_m', 'HDOP']):
            gt = gps_df['time_s'].to_numpy()
            gx = gps_df['x_m'].to_numpy()
            gy = gps_df['y_m'].to_numpy()
            gh = gps_df['HDOP'].to_numpy()
            tol = max(0.5, 2.0 * np.median(np.diff(gt)) if len(gt) > 1 else 1.0)
            for i, t in enumerate(time):
                idx = np.argmin(np.abs(gt - t))
                if np.abs(gt[idx] - t) < tol:
                    gps_aligned[i] = [gx[idx], gy[idx], gh[idx]]

    return {
        "seq_name": seq_name,
        "ronin_pos": ronin_pos,
        "gps_pos": gps_aligned[:, :2],
        "gps_hdop": gps_aligned[:, 2],
        "gt_pos": gt_pos,
        "time": time,
        "start_frame": start_frame,
    }

def load_aligned_sequences(seq_names, ronin_dir, data_root):
    """Load multiple sequences and compute velocities"""
    seqs = []
    for seq_name in tqdm(seq_names, desc="Loading sequences"):
        d = load_sequence_data(seq_name, ronin_dir, data_root)
        if d is None:
            continue
        dt = safe_dt(d['time'])
        rp = ffill_2d(d['ronin_pos'])
        ronin_vel = np.zeros_like(rp, dtype=float)
        if len(rp) > 1:
            ronin_vel[1:] = (rp[1:] - rp[:-1]) / dt[1:, None]
            ronin_vel[0] = ronin_vel[1]
        d['ronin_vel'] = ronin_vel
        seqs.append(d)
    return seqs
