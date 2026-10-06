"""
validation -- AxisForge checked against independent references.

Only ``validation._support`` is an importable package (reports, figures and
convergence adapters shared by the validation cases). The case folders
(``validation/<domain>/<case>/``) are not packages: each script imports its
own ``construction.py`` as a sibling module.
"""
