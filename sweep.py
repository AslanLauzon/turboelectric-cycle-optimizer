"""
Mach sweep along the climb schedule, plus baselines.

Optimized engine   all three modes, warm-started from the previous Mach.
Turbojet baseline  burner always lit: optimizer restricted to fan_burner.
Fixed-π_f baseline fan always on at one π_f; burner is the only throttle.
                   π_f = largest value the power limit allows at every Mach on
                   the schedule (set by the high-Mach end). When the fan alone
                   over-delivers, the burner is off and the excess is wasted.

Both baselines are restricted versions of the optimizer's search space, so
the optimized fuel flow can never exceed either.
"""

import numpy as np
from scipy.optimize import brentq

from optimizer import optimize_point, _point, _ok
from diagnostics import max_fan_thrust, P_LIM
from mission import climb_altitude, thrust_required
from config import T_BURNER_MAX, ETA_CHAIN

TRANSITION_TOL = 1e-3   # Mach resolution of bisected mode transitions


def _on_schedule(M):
    return float(M), float(climb_altitude(M))


def fixed_pi_design(M_grid=np.linspace(0.0, 3.0, 121)):
    """Largest constant π_f that is power-feasible everywhere on the schedule."""
    return min(max_fan_thrust(*_on_schedule(m))["pi_f"] for m in M_grid)


def fixed_pi_point(M, h, pi_f, F_req=None):
    """Fixed-π_f turbofan: fan at pi_f, burner throttled to meet F_req."""
    if F_req is None:
        F_req = float(thrust_required(M, h))
    r = _point(M, h, pi_f, None)
    if r["F"] < F_req:
        T25 = float(r["T25"])
        g = lambda T4: float(_point(M, h, pi_f, T4)["F"]) - F_req
        if T25 + 1.0 >= T_BURNER_MAX or g(T_BURNER_MAX) < 0:
            r = None
        else:
            lo = T25 + 1.0
            T4 = lo if g(lo) >= 0 else brentq(g, lo, T_BURNER_MAX, xtol=1e-9)
            r = _point(M, h, pi_f, T4)
    if r is None or not _ok(r, F_req):
        return {"M": M, "h": h, "F_req": F_req, "pi_f": pi_f, "T4": None,
                "P_gen": None, "F": None, "total_fuel": np.inf, "TSFC": np.inf}
    return {"M": M, "h": h, "F_req": F_req, "pi_f": pi_f,
            "T4": float(r["T4"]) if r["burner_on"] else None,
            "P_gen": float(r["P_gen"]), "F": float(r["F"]),
            "total_fuel": float(r["total_fuel"]),
            # fuel per required thrust: over-delivered thrust is not credited
            "TSFC": float(r["total_fuel"]) / F_req}


def mach_sweep(M_grid, modes=("fan_only", "fan_burner", "ram"), warm=True):
    """Optimize every point of M_grid on the climb schedule."""
    rows, x0 = [], None
    for m in M_grid:
        r = optimize_point(*_on_schedule(m), modes=modes, x0=x0 if warm else None)
        if r["x_fan_burner"] is not None:
            x0 = r["x_fan_burner"]
        rows.append(r)
    return rows


def power_down_onset(rows, frac=0.99):
    """
    First Mach after burner light-off where P_gen drops below frac·P_LIM,
    i.e. the fan starts powering down. None if it never does.
    """
    lit = False
    for r in rows:
        if r["mode"] == "fan_burner":
            if lit and r["P_gen"] * ETA_CHAIN < frac * P_LIM:
                return r["M"]
            lit = lit or r["P_gen"] * ETA_CHAIN >= frac * P_LIM
        elif lit and r["mode"] == "ram":
            return r["M"]
    return None


def find_transitions(rows, tol=TRANSITION_TOL):
    """
    Bisect each change of selected mode between adjacent sweep points.
    Returns list of (M_switch, mode_before, mode_after).
    """
    out = []
    for a, b in zip(rows, rows[1:]):
        if a["mode"] == b["mode"]:
            continue
        lo, hi, x0 = a["M"], b["M"], a["x_fan_burner"]
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            r = optimize_point(*_on_schedule(mid), x0=x0)
            if r["mode"] == a["mode"]:
                lo = mid
            else:
                hi = mid
        out.append((0.5 * (lo + hi), a["mode"], b["mode"]))
    return out
