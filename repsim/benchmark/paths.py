import os
from pathlib import Path

from loguru import logger


def get_experiments_path() -> Path:
    """
    Path to directory containing all (downloaded/trained) models used to extract representations, datasets, and results.
    Can be overridden by setting the environment variable 'REP_SIM'.
    Will contain subdirectories of `nlp`, `graph`, `vision`, and `results`.
    """
    try:
        EXPERIMENTS_ROOT_PATH = os.environ["REP_SIM"]  # To be renamed to ones liking
        return Path(EXPERIMENTS_ROOT_PATH)
    except KeyError:
        logger.warn("No 'DATA_RESULTS_FOLDER' Env variable -- Defaulting to '<project_root>/experiments' .")
        exp_pth = Path(__file__).parent.parent.parent / "experiments"
        exp_pth.mkdir(exist_ok=True)
        return exp_pth


BASE_PATH = get_experiments_path()
VISION_MODEL_PATH = Path(BASE_PATH, "models", "vision")
VISION_DATA_PATH = Path(BASE_PATH, "datasets", "vision")
VISION_TAR_PATH = Path(VISION_MODEL_PATH, "vissimbench.tar.gz")
EXPERIMENT_RESULTS_PATH = Path(BASE_PATH, "results")
