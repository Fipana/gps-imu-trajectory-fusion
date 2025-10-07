# GPS-Assisted Trajectory Prediction Using IMU Sensors

Deep learning approach for robust trajectory estimation by fusing GPS and IMU sensor data using a velocity correction LSTM model on top of the RoNIN baseline.

## Problem Statement

This project addresses trajectory prediction for wearable devices in environments where GPS signals may be weak, noisy, or intermittent. We combine:
- **IMU sensors** for continuous motion tracking
- **GPS data** for position correction when signals are strong
- **Deep learning** to predict velocity corrections

## Architecture
IMU Data → RoNIN ResNet → Velocity Predictions

↓

GPS Data → Velocity Correction LSTM → Corrected Velocities

↓

Fused Trajectory 


## Installation
```bash
git clone https://github.com/Fipana/gps-imu-trajectory-fusion.git
cd gps-imu-trajectory-fusion
pip install -r requirements.txt
```

## (Optional) set paths for your machine
```bash
export DATA_DIR=/mnt/datasets/frdr
export MODEL_DIR=./models
export PROJECT_DIR=./runs/exp01

```

## Usage

### Training
```bash
python scripts/train.py --config configs/train_config.yaml
```

### Evaluation
```bash
python scripts/evaluate.py \
  --model models/velocity_correction_model.pth \
  --ronin_dir ronin_predictions/unseen \
  --data_dir data/unseen_subjects_test_set \
  --split unseen \
  --output_dir results
```


## Project Structure
```bash
gps-imu-trajectory-fusion/
├─ src/
│  ├─ data/            # loaders + alignment
│  ├─ models/          # VC-LSTM
│  ├─ training/        # loss + trainer
│  ├─ inference/       # fusion logic
│  ├─ evaluation/      # metrics + eval loop
│  └─ paths.py         # path managers
├─ scripts/            # train.py, evaluate.py (use paths.py)
├─ configs/            # YAML (repo-relative paths)
├─ data/               # seen_subjects_test_set/, unseen_subjects_test_set
├─ ronin_predictions/  # seen/, unseen/  (npy baselines; gitignored)
├─ models/             # saved weights (gitignored)
├─ runs/               # results/plots/logs (gitignored)
├─ notebooks/
├─ requirements.txt
└─ .gitignore

```

## Key Features

- Velocity-level fusion for stable integration
- HDOP-aware weighting prevents trusting poor GPS
- GPS dropout augmentation for robustness
- Temporal decay strategy for extended dropouts

## LLM Usage Declaration
LLMs were used for:

- Code organization and modularization
- Documentation and formatting
- Debugging assistance

## References

- RoNIN: Herath et al., "Robust Neural Inertial Navigation in the Wild" (2020)
- Dataset: FRDR RoNIN Dataset with Synthetic GPS
