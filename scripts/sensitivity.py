"""
One-at-a-time sensitivity of the predicted mode transitions.

    uv run python scripts/sensitivity.py

Each case runs in a fresh subprocess that patches config before any model
module imports it (config values are bound at import). CDA_0 is recalibrated
at M 0.6 for every case except the CDA_0 case itself.
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_MACH = 61

CASES = [
    ("base", None, None),
    ("CDA_0", "CDA_0", 0.8 * 0.4470), ("CDA_0", "CDA_0", 1.2 * 0.4470),
    ("A_FAN", "A_FAN", 0.4), ("A_FAN", "A_FAN", 0.6),
    ("PSFC_GEN [kg/kWh]", "PSFC_GEN", 0.25), ("PSFC_GEN [kg/kWh]", "PSFC_GEN", 0.35),
    ("T_BURNER_MAX [K]", "T_BURNER_MAX", 1900.0), ("T_BURNER_MAX [K]", "T_BURNER_MAX", 2300.0),
    ("M_FACE", "M_FACE", 0.50), ("M_FACE", "M_FACE", 0.60),
]


def worker(param, value):
    """Runs in the subprocess: patch config, sweep, print JSON."""
    sys.path.insert(0, ROOT)
    import config
    if param is not None:
        setattr(config, param, value / 3.6e6 if param == "PSFC_GEN" else value)

    import numpy as np
    import mission
    from diagnostics import calibrate_cda0
    from sweep import mach_sweep, find_transitions, power_down_onset

    cda0 = config.CDA_0 if param == "CDA_0" else calibrate_cda0()
    # thrust_required binds CDA_0 as a default argument at import
    mission.thrust_required.__defaults__ = (cda0,)

    rows = mach_sweep(np.linspace(0.0, 3.0, N_MACH))
    trans = {(a, b): m for m, a, b in find_transitions(rows)}
    print(json.dumps({
        "cda0": cda0,
        "light_off": trans.get(("fan_only", "fan_burner")),
        "onset": power_down_onset(rows),
        "ram": trans.get(("fan_burner", "ram")),
    }))


def run_case(case):
    label, param, value = case
    code = (f"import sys; sys.path.insert(0, {os.path.join(ROOT, 'scripts')!r}); "
            f"import sensitivity; sensitivity.worker({param!r}, {value!r})")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True, cwd=ROOT)
    return case, json.loads(out.stdout.strip().splitlines()[-1])


def fmt(x):
    return "   -  " if x is None else f"{x:6.3f}"


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        results = list(ex.map(run_case, CASES))

    print(f"{'parameter':>18} {'value':>7} {'CDA_0':>6} | {'light-off':>9} "
          f"{'onset':>6} {'ram':>6}   (power-down onset at {3.0/(N_MACH-1):.3f} Mach resolution)")
    for (label, param, value), r in results:
        v = "-" if value is None else f"{value:.4g}"
        print(f"{label:>18} {v:>7} {r['cda0']:6.3f} | {fmt(r['light_off']):>9} "
              f"{fmt(r['onset'])} {fmt(r['ram'])}")
