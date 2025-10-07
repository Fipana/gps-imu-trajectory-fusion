"""Dataset for velocity correction training"""
import numpy as np
import torch
from torch.utils.data import Dataset
from ..utils.helpers import safe_dt, ffill_2d

class VelocityCorrectionDataset(Dataset):
    def __init__(self, sequences, window=200, stride=100, min_gps_cov=0.05,
                 augment_gps_dropout=True, dropout_prob=0.3):
        self.samples = []

        for seq in sequences:
            T = len(seq['time'])
            dt = safe_dt(seq['time'])

            # Compute target velocity corrections
            gt_vel = np.zeros_like(seq['gt_pos'], dtype=float)
            if T > 1:
                gt_vel[1:] = (seq['gt_pos'][1:] - seq['gt_pos'][:-1]) / dt[1:, None]
                gt_vel[0] = gt_vel[1]
            vel_corr = (gt_vel - seq['ronin_vel']).astype(np.float32)

            # Prepare GPS features
            gps_flag = np.isfinite(seq['gps_pos']).all(axis=1).astype(np.float32)
            gps_pos = ffill_2d(seq['gps_pos'])
            gps_pos = np.nan_to_num(gps_pos, nan=0.0)

            gps_vel = np.zeros_like(gps_pos)
            if T > 1:
                gps_vel[1:] = (gps_pos[1:] - gps_pos[:-1]) / dt[1:, None]
                gps_vel[0] = gps_vel[1]

            gps_pos *= gps_flag[:, None]
            gps_vel *= gps_flag[:, None]
            hdop = np.nan_to_num(seq['gps_hdop'], nan=10.0).clip(0, 20.0)

            # Build features: [ronin_vel(2), gps_pos(2), gps_vel(2), hdop(1), dt(1), flag(1)] = 9D
            X = np.column_stack([
                seq['ronin_vel'], gps_pos, gps_vel,
                hdop[:, None], dt[:, None], gps_flag[:, None]
            ]).astype(np.float32)
            M = gps_flag[:, None].astype(np.float32)

            # Create windows
            for i in range(0, T - window + 1, stride):
                if M[i:i+window].mean() < min_gps_cov:
                    continue
                Xw = X[i:i+window].copy()
                Yw = vel_corr[i:i+window].copy()
                Mw = M[i:i+window].copy()
                self.samples.append({'X': Xw, 'Y': Yw, 'M': Mw})

                # GPS dropout augmentation
                if augment_gps_dropout and np.random.rand() < dropout_prob:
                    X_aug = Xw.copy()
                    M_aug = Mw.copy()
                    dropout_len = np.random.randint(window // 5, int(window * 0.6))
                    s = np.random.randint(0, window - dropout_len)
                    e = s + dropout_len
                    X_aug[s:e, 2:6] = 0
                    X_aug[s:e, 6] = 10.0
                    X_aug[s:e, 8] = 0
                    M_aug[s:e] = 0
                    self.samples.append({'X': X_aug, 'Y': Yw, 'M': M_aug})

        print(f"Dataset: {len(self.samples)} windows")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        X = np.nan_to_num(s['X'], nan=0.0).astype(np.float32)
        Y = np.nan_to_num(s['Y'], nan=0.0).astype(np.float32)
        M = s['M'].astype(np.float32)
        return {
            'X': torch.tensor(X),
            'Y': torch.tensor(Y),
            'M': torch.tensor(M),
        }
