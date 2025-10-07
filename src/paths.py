
from pathlib import Path
import os, json

def _env_path(key: str, default: Path) -> Path:
    p = Path(os.getenv(key, str(default))).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p

def get_paths():
    # project root = repo root (…/gps-imu-trajectory-fusion)
    repo_root = Path(__file__).resolve().parents[1]
    defaults = {
        "DATA_DIR":    repo_root / "data",
        "MODEL_DIR":   repo_root / "models",
        "PROJECT_DIR": repo_root / "runs",  # where to put outputs/logs
    }
    return {k: _env_path(k, v) for k, v in defaults.items()}

def print_paths():
    print(json.dumps({k: str(v) for k,v in get_paths().items()}, indent=2))
