from pathlib import Path
import argparse
from utils.config import load_config
from utils.paths import p, ensure_dir
from utils.logging import get_logger

log = get_logger("download_kaggle")

def main() -> None:
    cfg = load_config()
    raw_dir = p(cfg["paths"]["data_raw"])
    ensure_dir(raw_dir)
    log.info("Raw data dir ensured at %s", raw_dir)
    log.info("Phase 1 will implement Kaggle download/extract here.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.parse_args()
    main()
