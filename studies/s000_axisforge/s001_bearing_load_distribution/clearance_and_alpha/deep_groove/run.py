"""Clearance and contact angle question for the deep groove ball bearing (locating; partner: roller)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # studies/bearings/load_distribution

from _common.bearings import DeepGrooveSpec                     # noqa: E402
from clearance_and_alpha.study import ClearanceAndAlphaStudy    # noqa: E402

if __name__ == "__main__":
    ClearanceAndAlphaStudy(DeepGrooveSpec()).run_and_save(HERE)
