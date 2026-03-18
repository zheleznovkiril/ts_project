from pathlib import Path

SEED = 42
DATASET_PATH = Path(__file__).parent / "data" / "m4_monthly_dataset.tsf" # файлик с данными надо положить в data/, но если другой путь надо указать полный путь до него
N_SERIES = 200