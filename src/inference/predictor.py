"""
Inference and fusion logic
"""
import numpy as np
import torch


def predict_sequence_with_corrections(model, seq_data, device,
                                     hdop_threshold=2.0, max_correction=0.15):
    """
    Generate fused trajectory using velocity corrections.

    Args:
        model: Trained velocity correction model
        seq_data: Sequence dictionary
        device: Device to run inference on
        hdop_threshold: HDOP threshold for good GPS
        max_correction: Maximum correction magnitude

    Returns:
        Tuple of (fused_pos, vel_corrections, gps_mask)
    """
    T = len(seq_data['time'])
    dt = _safe_dt(seq_data['time'])
    hdop = np.nan_to_num(seq_data['gps_hdop'], nan=10.0).clip(0, 20.0)
    gps_flag = np.isfinite(seq_data['gps_pos']).all(axis=1).astype(np.float32)

    # Prepare GPS inputs
    gps_pos_in = _ffill_2d(seq_data['gps_pos'])
    gps_pos_in = np.nan_to_num(gps_pos_in, nan=0.0)

    gps_vel_in = np.zeros_like(gps_pos_in)
    if T > 1:
        gps_vel_in[1:] = (gps_pos_in[1:] - gps_pos_in[:-1]) / dt[1:, None]
        gps_vel_in[0] = gps_vel_in[1]

    gps_pos_in *= gps_flag[:, None]
    gps_vel_in *= gps_flag[:, None]

    # Build features
    features = np.column_stack([
        seq_data['ronin_vel'],
        gps_pos_in,
        gps_vel_in,
        hdop[:, None],
        dt[:, None],
        gps_flag[:, None],
    ]).astype(np.float32)

    # Predict corrections
    model.eval()
    corrections = []
    chunk = 5000
    with torch.no_grad():
        for i in range(0, T, chunk):
            end = min(i + chunk, T)
            X = torch.from_numpy(features[i:end]).unsqueeze(0).to(device)
            corr = model(X).squeeze(0).cpu().numpy()
            corrections.append(corr)
    vel_corrections = np.vstack(corrections)

    # Clamp corrections
    corr_mag = np.linalg.norm(vel_corrections, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_correction / (corr_mag + 1e-9))
    vel_corrections = vel_corrections * scale

    # Fusion logic
    fused_pos = np.zeros_like(seq_data['gt_pos'], dtype=float)
    fused_pos[0] = seq_data['ronin_pos'][0]

    consecutive_no_gps = 0
    last_good_gps_idx = 0

    for i in range(1, T):
        gps_available = gps_flag[i] > 0.5
        gps_good = gps_available and hdop[i] < hdop_threshold

        if gps_good:
            # Use GPS directly
            fused_pos[i] = seq_data['gps_pos'][i]
            consecutive_no_gps = 0
            last_good_gps_idx = i
        elif gps_available and hdop[i] < 5.0:
            # Blend GPS with corrected IMU
            corrected_vel = seq_data['ronin_vel'][i] + 0.3 * vel_corrections[i]
            imu_pred = fused_pos[i-1] + corrected_vel * dt[i]
            fused_pos[i] = 0.7 * seq_data['gps_pos'][i] + 0.3 * imu_pred
            consecutive_no_gps = 0
        else:
            # Use corrected IMU with decay
            consecutive_no_gps += 1
            decay = np.exp(-consecutive_no_gps / 200.0)
            corrected_vel = seq_data['ronin_vel'][i] + decay * vel_corrections[i]
            fused_pos[i] = fused_pos[i-1] + corrected_vel * dt[i]

            # Fallback for long GPS dropouts
            if consecutive_no_gps > 500 and last_good_gps_idx > 0:
                ronin_drift = seq_data['ronin_pos'][i] - seq_data['ronin_pos'][last_good_gps_idx]
                fallback = seq_data['gps_pos'][last_good_gps_idx] + ronin_drift
                fused_pos[i] = 0.7 * fused_pos[i] + 0.3 * fallback

    return fused_pos, vel_corrections, (gps_flag > 0.5)


def _safe_dt(t):
    """Compute safe time deltas"""
    t = np.asarray(t, float)
    dt = np.diff(t, prepend=t[0])
    pos = dt[np.isfinite(dt) & (dt > 0)]
    fill = np.median(pos) if pos.size else 0.1
    bad = ~np.isfinite(dt) | (dt <= 0)
    dt[bad] = fill
    return dt


def _ffill_2d(arr):
    """Forward fill NaN values"""
    a = np.asarray(arr, float).copy()
    finite = np.isfinite(a).all(axis=1)
    if not finite.any():
        return np.nan_to_num(a, nan=0.0)
    last = np.where(finite)[0][0]
    if last > 0:
        a[:last] = a[last]
    for i in range(last + 1, len(a)):
        if not np.isfinite(a[i]).all():
            a[i] = a[i - 1]
    return a
