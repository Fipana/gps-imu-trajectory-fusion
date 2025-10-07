
# src/utils/resolve.py
from pathlib import Path
import os

def resolve_data_root(repo_root: Path, yaml_rel: str, cli_override: str | None, env_key: str | None = None) -> Path:
   
    if cli_override:
        return Path(cli_override).expanduser().resolve()
    if env_key and os.getenv(env_key):
        return Path(os.getenv(env_key)).expanduser().resolve()
    return (repo_root / yaml_rel).expanduser().resolve()

def resolve_ronin_dirs(repo_root: Path, yaml_seen_rel: str, yaml_unseen_rel: str,
                       cli_seen: str | None, cli_unseen: str | None,
                       base_env: str = "RONIN_DIR_BASE") -> dict:
  
    if cli_seen:
        seen = Path(cli_seen).expanduser().resolve()
    else:
        base = os.getenv(base_env)
        if base:
            seen = (Path(base) / "seen").expanduser().resolve()
        else:
            seen = (repo_root / yaml_seen_rel).expanduser().resolve()

    if cli_unseen:
        unseen = Path(cli_unseen).expanduser().resolve()
    else:
        base = os.getenv(base_env)
        if base:
            unseen = (Path(base) / "unseen").expanduser().resolve()
        else:
            unseen = (repo_root / yaml_unseen_rel).expanduser().resolve()

    return {"seen": seen, "unseen": unseen}
