from utils.config import load_config
from utils.logging import get_logger

log = get_logger("build_reports")

def main() -> None:
    _ = load_config()
    log.info("Report builder stub. Populates metrics/fairness CSV in later phases.")

if __name__ == "__main__":
    main()
