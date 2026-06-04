from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUT_DIR = PROJECT_DIR / "outputs"

DEFAULT_DATA_PATH = RAW_DATA_DIR / "creditcard.csv"
RANDOM_STATE = 42
