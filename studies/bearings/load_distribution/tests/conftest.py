"""Make ``_common`` and the question packages importable from the family folder."""
import sys
from pathlib import Path

FAMILY_DIR = Path(__file__).resolve().parents[1]      # studies/bearings/load_distribution
if str(FAMILY_DIR) not in sys.path:
    sys.path.insert(0, str(FAMILY_DIR))
