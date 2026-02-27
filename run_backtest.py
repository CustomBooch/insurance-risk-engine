import os
import logging
from datetime import datetime

from data.synthetic_generator import generate_synthetic_panel
from backtest import expanding_window_backtest

# -------------------------------
# Logging Configuration
# -------------------------------

os.makedirs("logs", exist_ok=True)

log_file = os.path.join("logs", "backtest_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="a"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# -------------------------------
# Run Backtest
# -------------------------------

logger.info("=" * 60)
logger.info("New Backtest Run")
logger.info("=" * 60)

df = generate_synthetic_panel()

results = expanding_window_backtest(df)

logger.info("\nBacktest Summary:")
logger.info(f"\n{results}")

logger.info(f"\nAverage Frequency GLM Deviance: {results['glm_freq_dev'].mean()}")
logger.info(f"Average Frequency LGBM Deviance: {results['lgbm_freq_dev'].mean()}")

logger.info(f"\nAverage Severity GLM Deviance: {results['glm_sev_dev'].mean()}")
logger.info(f"Average Severity LGBM Deviance: {results['lgbm_sev_dev'].mean()}")