import os
from warnings import warn
from pathlib import Path


def get_experiments_path() -> str:
    """
    Path containing all (downloaded/trained) Models used to extract representations.
    Can be overridden by setting the environment variable 'REP_SIM'.
    Will contain subdirectories of `nlp`, `graph` and `vision`.
    """
    try:
        EXPERIMENTS_ROOT_PATH = os.environ["REP_SIM"]  # To be renamed to ones liking
        return EXPERIMENTS_ROOT_PATH
    except KeyError:
        warn("Could not find 'DATA_RESULTS_FOLDER' -- Defaulting to '<project_root>/experiments' .")
        exp_pth = Path(__file__).parent / "experiments"
        exp_pth.mkdir(exist_ok=True)
        return exp_pth
