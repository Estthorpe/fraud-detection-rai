from pathlib import Path
import os
import yaml
from dotenv import load_dotenv
from utils.paths import p  # <-- absolute import

load_dotenv()

def load_config(path: Path | None = None) -> dict:
    cfg_path = path or p("config", "default.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)
