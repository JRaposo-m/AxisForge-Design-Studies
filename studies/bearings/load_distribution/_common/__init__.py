"""
Shared code of the ``studies/bearings/load_distribution`` family.

Every study of this family uses the same two-shaft gear system and the same ISO/TS 16281
solve; only the bearing under study, the swept axes and the load change. This package holds
that common part ONCE:

    bearings/   one specification class per bearing type (geometry only, no data)
    system.py   SystemSpec and the two-shaft gear system (built once, not solved)
    solve.py    shaft FEM and the ISO/TS 16281 solve of one bearing
    rows.py     the result table: one row per (point of the sweep, shaft)
    study.py    LoadDistributionStudy, the sweep loop shared by every question
    reports.py  report.txt and the CSV tables (system, FEM, ISO/TS 16281) of each bearing type
    store.py    writes and reads ``<question>/<type>/results``
    plots.py    the figure style and the generic figures
    compare.py  reads the data of the bearing types of ONE question and compares them

This package calls the AxisForge API directly; it does not hide it behind a layer of its own.
It never stores results: those live in ``<question>/<type>/results``.
"""
