# GPS-Assisted Trajectory Prediction Using IMU Sensors

Deep learning approach for robust trajectory estimation by fusing GPS and IMU sensor data using a velocity-correction LSTM model on top of the RoNIN baseline.

## Overview

We:

- run RoNIN to get baseline 2D positions (*_gsn.npy),

- train a Velocity Correction LSTM (VC-LSTM) that predicts velocity corrections using GPS-(HDOP) and availability,

- fuse corrected IMU velocity with GPS to produce a robust trajectory.

- Both training and evaluation are driven by a single YAML config (configs/train_config.yaml). 

## Architecture
IMU Data → RoNIN ResNet → Velocity Predictions

↓

GPS Data → Velocity Correction LSTM → Corrected Velocities

↓

Fused Trajectory


## Google Colab notebook (full pipeline) --> Recommneded.

A Colab notebook covering the entire pipeline (data prep, RoNIN inference, training, evaluation, and visualization) is included [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/189J9vjs-mYSZS0rgzj6gAbn9Mhkw4efY?usp=sharing).


## Datasets & pretrained assets (Google Drive)

The modified dataset, the pretrained RoNIN baseline, and our fusion model artifacts (checkpoints, configs, logs) are available in a shared Google Drive folder:
GPSRoNIN project — https://drive.google.com/drive/folders/1I_6yqpJD3aoogOKeUoDuoim0Xyxv3U_-?usp=drive_link


## Installation
```bash
git clone https://github.com/Fipana/gps-imu-trajectory-fusion.git
cd gps-imu-trajectory-fusion
python -m venv .venv
# Linux/Mac:
. .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

```

## (Optional) set paths
```bash
export DATA_DIR=/GPSRoNIN/FRDR_dataset/Data
export MODEL_DIR=/GPSRoNIN/FRDR_dataset/Pretrained_Models
export PROJECT_DIR=/GPSRoNIN/gps-imu-project

```

## Usage

### Training
```bash

python scripts/train.py   --config configs/train_config.yaml   --project_dir "/ABS/PATH/TO/gps-imu-project"   --data_dir "/ABS/PATH/TO/seen_subjects_test_set"   --output "models/velocity_correction_model.pth"
```

### Evaluation
```bash
python scripts/evaluate.py   --config configs/train_config.yaml   --model "models/velocity_correction_model.pth"   --split unseen   --project_dir "/ABS/PATH/TO/gps-imu-project"   --data_dir "/ABS/PATH/TO/unseen_subjects_test_set"   --output_dir "results"

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
│  └─ paths.py
├─ scripts/            # train.py, evaluate.py
├─ configs/            # YAML
├─ ronin_predictions/  # seen/, unseen/
├─ models/             # weights (gitignored)
├─ runs/               # results/plots/logs (gitignored)
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
