"""Dry normal contact of two elastic bodies, in four levels imported in this order:
``core`` (the problem: bodies, contact, machine elements) -> ``mesh`` (the grid) -> ``results``
(data only) -> ``solvers`` (Hertz and SlipPY BEM). A level imports only the levels before it."""
