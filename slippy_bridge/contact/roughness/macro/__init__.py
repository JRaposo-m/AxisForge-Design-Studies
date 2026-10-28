"""Rough surfaces on the curved bodies of a contact (the macro scale): constructions only.

    spec       RoughnessSpec: ACF, PSD or measured roughness, its S- and L-filter, seed, composite
    surfaces   rough_heights (heights on a grid, grid-rule check), rough_surface_pair (surface_pair
               plus the heights), band_limit
    measures   rough_measures on a pressure field given by the caller (real area, patches, p_max)

``solvers.solve`` does not take roughness yet (later run): a caller builds the surfaces here and
drives SlipPY ``ContactModel`` + ``StaticStep`` itself.
"""
from .measures import THRESHOLD_REL, RoughMeasures, rough_measures
from .spec import RoughnessSpec
from .surfaces import (BETA_OVER_H_MIN, LAMBDA_S_OVER_H_MIN, RoughHeights, band_limit, rough_heights,
                       rough_surface_pair)

__all__ = ["RoughnessSpec", "RoughHeights", "rough_heights", "rough_surface_pair", "band_limit",
           "LAMBDA_S_OVER_H_MIN", "BETA_OVER_H_MIN", "RoughMeasures", "rough_measures", "THRESHOLD_REL"]
