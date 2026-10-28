"""s104 tip edge zone: BEM line contact of the FZG C40 flanks next to a sharp tip near A and E, against Hertz (README)."""
from .construction import contact_at, contacts_near_end, edge_of, fzg_c40
from .model import grid_series, hertz_measures, measures

__all__ = ["fzg_c40", "contact_at", "contacts_near_end", "edge_of", "measures", "hertz_measures", "grid_series"]
