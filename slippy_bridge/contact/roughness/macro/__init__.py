"""Rough surfaces on the curved bodies of a contact (the macro scale): constructions and measures.

    spec       RoughnessSpec: ACF, PSD or measured roughness, its S- and L-filter, seed, composite
    surfaces   rough_heights (heights on a grid, grid-rule check), band_limit
    measures   rough_measures on a pressure field given by the caller (real area, patches, p_max)

The heights go to the solver: ``solvers.solve(contact, grid, output_dir, heights=(h1, h2))`` (or
``solvers.surface_pair(..., heights=...)``) adds them to the body profiles; ``rough_measures`` then
reads ``StaticResult.p``.
"""
from .measures import THRESHOLD_REL, RoughMeasures, rough_measures
from .spec import RoughnessSpec
from .surfaces import BETA_OVER_H_MIN, LAMBDA_S_OVER_H_MIN, RoughHeights, band_limit, rough_heights

__all__ = ["RoughnessSpec", "RoughHeights", "rough_heights", "band_limit",
           "LAMBDA_S_OVER_H_MIN", "BETA_OVER_H_MIN", "RoughMeasures", "rough_measures", "THRESHOLD_REL"]
