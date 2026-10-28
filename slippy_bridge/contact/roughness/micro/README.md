# roughness/micro: the periodic rough patch

**What it is for.** The micro scale gives contact laws of a rough surface against a flat, as
functions of the nominal pressure `p_nom` on a periodic cell of side `L`: the real contact area
fraction `A_real / L^2`, the mean gap `g_mean`, the normal contact stiffness `-d p_nom / d g_mean`
[Pa/m] (positive: the pressure rises as the gap closes), later the asperity load share in mixed EHL
(the fraction of the load carried by solid contact against the film pressure, the load sharing of
Johnson, Greenwood and Poon 1972: source to get). SI units; results scale with `p / E*`
(`PatchSpec.e_star`), the small-slope frictionless normal contact of Muser 2017 (section 2.1). A macro solve (a bearing or gear contact) or a lubrication model then reads these
laws instead of resolving every asperity.

**What is here (constructions only).** `patch.py`: `PatchSpec` (side `L`, `n` nodes per axis,
`RoughnessSpec`, the two materials, the nominal pressure range of the later sweep) and
`periodic_patch`, which returns the two SlipPY surfaces, periodic in both axes, each with
`Elastic(..., zero_frequency_value=0)` as SlipPY's `rey` solver requires
(`slippy/contact/_step_utils.py:352-353`, SB-61), and the heights record; the RVE checks warn
(`L / lambda_largest >= 5`, the challenge's ratio; the grid rule of `macro`).

**Boundary with the later run.** `patch.py` builds the patch and solves nothing. The law of the
patch goes in `law.py` here (later run, proposal of the R4b architect review): `contact_law(patch,
p_nom_Pa, output_dir) -> ContactLaw` (`p_nom_Pa`, `area_fraction`, `g_mean_m`,
`stiffness_Pa_per_m`, checks, commits), a fully periodic `StaticStep` (`periodic_geometry=True`,
`periodic_axes=(True, True)`, `method="rey"`) under load control `p_nom L^2` or `mean_gap`,
measured with `macro.rough_measures` (which needs a wrap-around connectivity for patches across the
periodic boundary). Under `rey` `loads_z` holds tiny positive values on almost every node (99.9 % with
`loads_z > 0` against 0.1 % in `contact_nodes`, slippy-expert R4b scratch check): `A_real` is taken from
`contact_nodes` or a relative threshold, never from `loads_z > 0`, and SlipPY's `interference` (an
average over `pressure > 0`, `_step_utils.py:366-367`) is not used. `normal_load` is a total force
[N] (`rey` divides it by `n^2 h^2`, `_step_utils.py:360`). Nothing in `solvers/` knows the patch: `solve` / `surface_pair` rebuild the
materials with the default zero-frequency term (a trap: OPEN R4-11). Consumers (macro corrections,
`lubrication/`) read `ContactLaw`, never SlipPY objects. There is no absolute approach, only the
mean gap.

**References to get / read.** Muser et al. 2017, the contact-mechanics challenge (`Muser2017`,
the judge: periodic self-affine surface, `A(p)`, gap distribution; V-RGH-01/02); Persson 2001
(`Persson2001`, area-load law, Eqs. (31)-(35)); Yastrebov et al. (2015, 2017) on the RVE size and
the area-pressure law of self-affine surfaces (not in the library: to get); Greenwood-Williamson
1966 (`GreenwoodWilliamson1966`) as the asperity-model context.
