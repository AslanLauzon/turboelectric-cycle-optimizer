"""
Spot check of the point optimizer along the climb schedule.

    uv run python scripts/optimize_points.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer import optimize_point, MODES
from mission import climb_altitude


def fmt(x, spec):
    return "-" if x is None else format(x, spec)


if __name__ == "__main__":
    print(f"{'M':>4} {'h':>6} {'F_req':>6} {'mode':>10} {'pi_f':>6} {'T4':>6} "
          f"{'P_gen':>5} {'fuel':>6} {'TSFC':>6} | "
          + " ".join(f"{m:>10}" for m in MODES) + f" {'ref':>4} {'t[s]':>5}")
    print(f"{'':>4} {'[km]':>6} {'[kN]':>6} {'':>10} {'':>6} {'[K]':>6} "
          f"{'[MW]':>5} {'[kg/s]':>6} {'kg/N/h':>6} | "
          + " ".join(f"{'[kg/s]':>10}" for _ in MODES))
    for M in (0.2, 0.6, 0.7, 0.8, 1.05, 1.5, 2.0, 2.5, 3.0):
        h = float(climb_altitude(M))
        t0 = time.perf_counter()
        r = optimize_point(M, h)
        dt = time.perf_counter() - t0
        bf = " ".join(f"{r['branch_fuel'][m]:10.4f}" for m in MODES)
        print(f"{M:4.2f} {h/1e3:6.1f} {r['F_req']/1e3:6.2f} {str(r['mode']):>10} "
              f"{fmt(r['pi_f'], '6.3f')} {fmt(r['T4'], '6.0f')} "
              f"{fmt(r['P_gen'] and r['P_gen']/1e6, '5.2f')} {r['total_fuel']:6.4f} "
              f"{r['TSFC']*3600:6.4f} | {bf} {str(r['refined'])[0]:>4} {dt:5.2f}")
