from utils.config import load_config
from utils.logging import get_logger

log = get_logger("train")

def main() -> None:
    cfg = load_config()
    log.info("Loaded config. Baseline model: %s", cfg["baseline"]["model"])
    log.info("This will be implemented in Phase 2 & 3.")

if __name__ == "__main__":
    main()
