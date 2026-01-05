# paths.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# TEST_DATA_DIR = PROJECT_ROOT / "data" / "test_data"
# TRAIN_DATA_DIR = PROJECT_ROOT / "data" / "train_data"
LLAMAF_DIR = PROJECT_ROOT / "LLaMA-Factory"
# SRC_DIR = PROJECT_ROOT / "src"

print("PROJECT_ROOT:", PROJECT_ROOT)
# print("TEST_DATA_DIR:", TEST_DATA_DIR)
# print("TRAIN_DATA_DIR:", TRAIN_DATA_DIR)
print("LLAMAF_DIR:", LLAMAF_DIR)
# print("SRC_DIR:", SRC_DIR)