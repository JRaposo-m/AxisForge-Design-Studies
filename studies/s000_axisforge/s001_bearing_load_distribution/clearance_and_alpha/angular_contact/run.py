"""
Clearance and contact angle question for the angular contact ball bearing (locating; partner: roller).

With spur gears (Fa = 0) a single-row angular contact bearing cannot reach axial equilibrium
under a radial load: see the warning in ../study.py. SystemSpec warns about it on purpose.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # studies/bearings/load_distribution

from _common.bearings import AngularContactSpec                 # noqa: E402
from clearance_and_alpha.study import ClearanceAndAlphaStudy    # noqa: E402

if __name__ == "__main__":
    ClearanceAndAlphaStudy(AngularContactSpec()).run_and_save(HERE)
