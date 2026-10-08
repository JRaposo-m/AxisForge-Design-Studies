"""
Clearance and contact angle question: the three types side by side (reads their results/ only).

Each type sweeps its own parameter, so the figures use RESULT columns every type reports:
    vs s_mm         deep groove, cylindrical roller and angular contact
    vs alpha0_deg   the two ball bearings (a roller has no contact angle)
"""
import sys
from pathlib import Path

QUESTION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QUESTION_DIR.parent))      # studies/bearings/load_distribution

from _common.compare import compare_types         # noqa: E402

TYPES = ["deep_groove", "cylindrical_roller", "angular_contact"]

if __name__ == "__main__":
    compare_types(QUESTION_DIR, TYPES, x_columns=["s_mm", "alpha0_deg"],
                  figure_types={"alpha0_deg": ["deep_groove", "angular_contact"]})
