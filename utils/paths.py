from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def root() -> Path:
    return PROJECT_ROOT

def p(*parts) -> Path:
    return PROJECT_ROOT.joinpath(*parts)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
