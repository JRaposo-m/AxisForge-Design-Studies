"""Run every build of every type of ``hertz_vs_bem``, in series, each in its own process.

Each run.py is a separate process because SlipPy requires unique material names per process and
because timings must not be disturbed by a previous run. A failing run does not stop the others.

Usage::

    python run_all.py                        # everything
    python run_all.py --type ball_raceway    # one type
    python run_all.py --build composite      # one build, all types
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

RUNS = [
    ("ball_flat", "reference"),
    ("ball_raceway", "composite"),
    ("ball_raceway", "two_profiles"),
    ("roller_raceway", "composite"),
    ("roller_raceway", "two_profiles"),
]

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--type", help="run only this type")
    parser.add_argument("--build", help="run only this build")
    args = parser.parse_args()

    selected = [(t, b) for t, b in RUNS
                if (args.type in (None, t)) and (args.build in (None, b))]
    results = []
    for type_name, build in selected:
        script = HERE / type_name / build / "run.py"
        print(f"\n===== {type_name} / {build}", flush=True)
        t0 = time.perf_counter()
        code = subprocess.run([sys.executable, str(script)], cwd=script.parent).returncode
        results.append((type_name, build, code, time.perf_counter() - t0))

    print("\n===== summary")
    for type_name, build, code, dt in results:
        print(f"  {'ok  ' if code == 0 else 'FAIL'}  {type_name:14s} {build:14s} {dt:7.1f} s")
    failed = [r for r in results if r[2] != 0]
    if failed:
        print("\nsome runs failed: compare/ must not be run on incomplete data")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
