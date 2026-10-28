"""The published systems v003 knows (D-0036 amendment item 4).

* Hamrock (1983), NASA RP-1105, sec. 14.2 (PDF p. 44-46): a deep groove ball bearing (9 balls of 12.7 mm,
  pitch diameter 65 mm, groove radii 0.52 d, diametral clearance 15 um, steel E 2e11 Pa, nu 0.3) under
  F_r = 8900 N; printed F_max = 4513 N and the compressions of that ball (delta_i, delta_o, delta =
  5.438e-5 m). An AxisForge system: :func:`hamrock_dgbb` builds it. Hamrock reaches F_max by iterating
  Stribeck's factor Z from an approximate fit (his Eq. (30)) with the Hamrock-Brewe contact fits, and his
  printed k_i, E_i, F_i, E_o, F_o do not follow from his own Eqs (37)-(39) (only k_o does; OPEN R5-09): the
  tolerances of README section 6 (L1, L2) are set from those two approximations, not from his digits.
* Hernot, Sartor, Guillot (2000), sec. 2.3 and Table 1 (PDF p. 3): a 218 angular contact ball bearing,
  Z = 16, k = 423000 N/mm^1.5 (both contacts), n = 1.5, alpha = 40 deg, F_r = F_a; the converged displacements
  and Q_max = 6544 N. The geometry is not printed, so it is NOT an AxisForge system: it judges r03 only (the
  judge of this case; its continuous branch), :func:`hernot_218` returns r03's inputs. The loads are printed
  178000 N (kept so in the CSV); Q_max and the displacements are those of 17800 N: :func:`hernot_218` applies
  the factor 1/10 (FLAG, OPEN R5-13).
* Harris2007_vol1 Fig. 7.3 (PDF p. 160) and Fig. 7.6 (PDF p. 163) are charts: not digitised, not used.

The rows, with their pages, are in ``published/*.csv`` (read from the page images, 2026-10-28; library notes
``06_Library/bearings/Hamrock1983.md`` and ``Hernot2000.md`` have no content pass yet). Hamrock: PDF page =
printed page + 5; Hernot: journal page = PDF page + 82.

Units: Hamrock and Hernot print SI (Hernot mm for displacements and k); converted here, once, to AxisForge's
mm, N, MPa (Hamrock) or to r03's SI (Hernot).
"""
from __future__ import annotations

import math
from pathlib import Path

from axisforge.outputs._format import read_records_csv

# OPEN R5-10 (as R2-02, R5-22): LiteratureCase is defined in v101 and imported until a common place is ruled
from studies.v100_dry_contact.v101_hertz_point.literature import LiteratureCase

from ..construction import deep_groove_6210

__all__ = ["cases", "hamrock_dgbb", "hernot_218", "published", "LiteratureCase", "PUBLISHED"]

PUBLISHED = Path(__file__).with_name("published")


def published(name: str) -> dict[str, dict]:
    """The rows of ``published/<name>.csv`` by symbol, values as floats where they are numbers."""
    rows = read_records_csv(PUBLISHED / f"{name}.csv")
    return {r["symbol"]: {**r, "value": float(r["value"])} for r in rows}


def hamrock_dgbb():
    """Hamrock 1983 sec. 14.2 bearing as an AxisForge 6210-family bearing, not solved.

    Pitch diameter (d_i + d_o)/2 = 0.065 m, clearance d_o - d_i - 2d = 1.5e-5 m (diametral, AxisForge ``s``),
    E 2e11 Pa = 200000 MPa. Ball diameter, ball count and groove radii (0.52 d, the family default) are those of
    construction. Boundary dimensions (d, D, B) are not printed: the 6210 ones are kept; the ISO/TS 16281 load
    distribution does not read them.
    """
    p = published("hamrock1983_sec14_2")
    d_i, d_o, d = p["d_i"]["value"], p["d_o"]["value"], p["d"]["value"]
    assert math.isclose(p["r_i"]["value"] / d, 0.52, rel_tol=1e-3)            # the family default f = 0.52
    return deep_groove_6210((d_o - d_i - 2.0 * d) * 1e3, Dpw_mm=(d_i + d_o) / 2.0 * 1e3,
                            E_MPa=p["E"]["value"] * 1e-6, label="hamrock1983")


def hernot_218() -> dict:
    """r03 inputs of Hernot's 218 bearing (SI): ``Z``, ``K_n`` [N/m^1.5], ``n``, ``alpha_rad``, ``Fr_N``, ``Fa_N``."""
    p = published("hernot2000_table1")
    # OPEN R5-13: printed 178000 N; Q_max 6544 N, Table 1 and Hernot's own Eqs (5), (8), (9) belong to 17800 N
    # (librarian check of v003): the correction is applied here, the published file keeps the print
    load_correction = 0.1
    return {"Z": int(p["Z"]["value"]), "K_n": p["k"]["value"] * 1e3 ** 1.5, "n": p["n"]["value"],
            "alpha_rad": math.radians(p["alpha"]["value"]), "Fr_N": p["F_r"]["value"] * load_correction,
            "Fa_N": p["F_a"]["value"] * load_correction}


def cases() -> list[LiteratureCase]:
    """The published systems of v003: Hamrock 1983 sec. 14.2 (AxisForge) and Hernot 2000 Table 1 (r03 only)."""
    return [LiteratureCase("hamrock1983_sec14_2_dgbb", hamrock_dgbb,
                           read_records_csv(PUBLISHED / "hamrock1983_sec14_2.csv"),
                           "Hamrock1983 sec. 14.2, Eqs (1)-(3), (30), (37)-(39), (42), (57), PDF p. 44-46"),
            LiteratureCase("hernot2000_218_acbb", hernot_218,
                           read_records_csv(PUBLISHED / "hernot2000_table1.csv"),
                           "Hernot2000 sec. 2.3 and Table 1, Eqs (1)-(9), PDF p. 2-3")]
