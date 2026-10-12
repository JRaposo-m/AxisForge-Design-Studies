"""3D views of AxisForge results with PyVista (plots live in design-studies, D-0008).

``viz.scene``: the generic part (named blocks, colour limits, window and PNG from one drawing
routine). The scene builders: ``viz.bearing`` (rings, balls, Hertz contact ellipses, applied load),
``viz.contact`` (the pressure of one contact against Hertz), ``viz.shaft`` (deflected shaft,
bending stress). Import the submodule you need: ``viz.bearing`` and ``viz.contact`` pull
``slippy_bridge`` (and SlipPY); ``viz.scene`` and ``viz.shaft`` do not.
"""
