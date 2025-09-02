from pathlib import Path
import subprocess
import sys
import argparse
from utils.config import load_config
from utils.paths import p, ensure_dir
from utils.logging import get_logger

log = get_logger("download_kaggle")

DATASET = "mlg-ulb/creditcardfraud"

def run(cmd: list[str]) -> None:
    log.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log.info(proc.stdout)
    if proc.returncode != 0:
        log.error("Command failed with exit code %d", proc.returncode)
        sys.exit(proc.returncode)


def main() -> None:
    cfg = load_config()
    raw_dir = p(cfg["paths"]["data_raw"])
    ensure_dir(raw_dir)
    
    zip_path = raw_dir / "creditcardfraud.zip"
    csv_path = raw_dir / "creditcard.csv"

    if csv_path.exists():
        log.info("Dataset already present at %s", csv_path)
        return

    # 1) Download
    run(["kaggle", "datasets", "download", "-d", DATASET, "-p", str(raw_dir)])
    # 2) Unzip (cross-platform: use python -m zipfile for portability)
    if not zip_path.exists():
        # Fallback find
        candidates = list(raw_dir.glob("*.zip"))
        if candidates:
            zip_path = candidates[0]
    if not zip_path.exists():
        log.error("Zip not found in %s", raw_dir)
        sys.exit(1)

    run([sys.executable, "-m", "zipfile", "-e", str(zip_path), str(raw_dir)])
    if not csv_path.exists():
        log.error("creditcard.csv not found after extraction")
        sys.exit(1)
    log.info("Downloaded & extracted to %s", csv_path)

if __name__ == "__main__":
    main()