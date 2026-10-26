"""Misalignment question: deep groove ball vs cylindrical roller (reads the results/ of both types)."""
import sys
from pathlib import Path

QUESTION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QUESTION_DIR.parent))      # studies/bearings/load_distribution

from _common.compare import compare_types         # noqa: E402

if __name__ == "__main__":
    compare_types(QUESTION_DIR, ["deep_groove", "cylindrical_roller"])
