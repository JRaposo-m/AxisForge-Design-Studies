"""Misalignment question for the cylindrical roller bearing (non-locating; partner: deep groove ball)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # studies/bearings/load_distribution

from _common.bearings import CylindricalRollerSpec   # noqa: E402
from misalignment.study import MisalignmentStudy     # noqa: E402

if __name__ == "__main__":
    MisalignmentStudy(CylindricalRollerSpec()).run_and_save(HERE)
