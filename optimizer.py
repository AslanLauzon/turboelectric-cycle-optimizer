"""
Point optimizer: minimum total fuel flow at one flight condition.

    min  ṁ_f,gen + ṁ_f,burner
    s.t. F ≥ F_req, and every propulsor_point constraint flag clear

P_gen is eliminated exactly: generator fuel rises with P_gen and thrust does
not depend on it, so the optimum is always P_gen = P_fan / η_chain.

Three modes are solved separately; the lowest-fuel feasible one wins:

    fan_only    π_f ∈ [PI_F_MIN, π_f,max], burner off      brentq on F(π_f) = F_req
    fan_burner  π_f ∈ [PI_F_MIN, PI_F_MAX], T4 > T25        grid, then SLSQP
    ram         π_f = 1 (fan off, P_gen = 0), T4 > T02      brentq on F(T4) = F_req

Splitting the modes keeps SLSQP off the burner on/off discontinuity (the
burner ΔP appears as soon as T4 > T25) and makes fan power-down a discrete
choice. The fan-off duct is lossless at π_f = 1 (optimistic; step 7).
"""

import numpy as np
from scipy.optimize import brentq, minimize

from propulsor import propulsor_point
from diagnostics import max_fan_thrust, P_LIM
from mission import thrust_required
from config import (
    ETA_CHAIN, PI_F_MIN, PI_F_MAX, T_BURNER_MAX, T_FAN_EXIT_MAX,
)

# Numerical settings (not physics assumptions)
N_GRID_PI = 30
N_GRID_T4 = 30
N_COARSE = 10            # per-axis grid for the second start on warm-started points
T4_GRID_MIN = 300.0      # K
F_RTOL = 1e-6            # accepted relative shortfall on F ≥ F_req
EDGE = 1e-9              # relative margin kept inside the power and T25 limits

MODES = ("fan_only", "fan_burner", "ram")


def _point(M, h, pi_f, T4):
    """propulsor_point with P_gen at the minimum that powers the fan."""
    if pi_f == 1.0:
        P_gen = 0.0
    else:
        P_fan = float(propulsor_point(M, h, pi_f, None, 0.0)["P_fan"])
        P_gen = P_fan / ETA_CHAIN * (1.0 + 1e-12)   # round-off margin on power_deficit
    return propulsor_point(M, h, pi_f, T4, P_gen)


def _ok(r, F_req):
    return bool(r["feasible"]) and r["F"] >= F_req * (1.0 - F_RTOL)


def _fan_only(M, h, F_req):
    pi_max = max_fan_thrust(M, h)["pi_f"]
    if pi_max < PI_F_MIN:
        return None
    g = lambda pi: float(_point(M, h, pi, None)["F"]) - F_req
    if g(pi_max) < 0:
        return None
    pi = PI_F_MIN if g(PI_F_MIN) >= 0 else brentq(g, PI_F_MIN, pi_max, xtol=1e-12)
    r = _point(M, h, pi, None)
    return r if _ok(r, F_req) else None


def _ram(M, h, F_req):
    T02 = float(propulsor_point(M, h, 1.0, None, 0.0)["T02"])
    lo = T02 + 1.0
    if lo >= T_BURNER_MAX:
        return None
    g = lambda T4: float(_point(M, h, 1.0, T4)["F"]) - F_req
    if g(T_BURNER_MAX) < 0:
        return None
    T4 = lo if g(lo) >= 0 else brentq(g, lo, T_BURNER_MAX, xtol=1e-9)
    r = _point(M, h, 1.0, T4)
    return r if _ok(r, F_req) else None


def _fan_burner_grid(M, h, F_req, n_pi=N_GRID_PI, n_t4=N_GRID_T4):
    best = None
    for pi in np.linspace(PI_F_MIN, PI_F_MAX, n_pi):
        T25 = float(_point(M, h, pi, None)["T25"])
        for T4 in np.linspace(T4_GRID_MIN, T_BURNER_MAX, n_t4):
            if T4 <= T25 + 1.0:
                continue
            r = _point(M, h, pi, T4)
            if _ok(r, F_req) and (best is None or r["total_fuel"] < best[2]["total_fuel"]):
                best = (pi, T4, r)
    return best


def _slsqp(M, h, F_req, pi0, T40):
    """SLSQP refinement of fan_burner from (pi0, T40). Returns result or None."""
    def ev(x):
        return _point(M, h, float(x[0]), 1000.0 * float(x[1]))

    cons = [
        {"type": "ineq", "fun": lambda x: ev(x)["F"] / F_req - 1.0},
        {"type": "ineq", "fun": lambda x: 1.0 - EDGE - ev(x)["P_fan"] / P_LIM},
        {"type": "ineq", "fun": lambda x: 1.0 - EDGE - ev(x)["T25"] / T_FAN_EXIT_MAX},
        {"type": "ineq", "fun": lambda x: (1000.0 * x[1] - ev(x)["T25"] - 1.0) / 1000.0},
    ]
    res = minimize(
        lambda x: float(ev(x)["total_fuel"]),
        x0=[pi0, T40 / 1000.0],
        method="SLSQP",
        bounds=[(PI_F_MIN, PI_F_MAX), (T4_GRID_MIN / 1000.0, T_BURNER_MAX / 1000.0)],
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 200},
    )
    r = ev(res.x)
    return r if (_ok(r, F_req) and r["burner_on"]) else None


def _fan_burner(M, h, F_req, x0=None):
    """
    Returns (result or None, refined, grid_fuel).

    With a warm start x0 = (π_f, T4), SLSQP runs from x0 and from the best
    point of a coarse grid, and the cheaper result wins: the fan_burner
    problem has a second local minimum (interior vs. power-limit corner), so
    x0 alone can land in the wrong basin when the Mach step is large.
    grid_fuel is nan in that case. Falls back to the full cold grid if both fail.
    """
    if x0 is not None:
        cands = [_slsqp(M, h, F_req, *x0)]
        coarse = _fan_burner_grid(M, h, F_req, N_COARSE, N_COARSE)
        if coarse is not None:
            cands.append(_slsqp(M, h, F_req, coarse[0], coarse[1]))
        cands = [r for r in cands if r is not None]
        if cands:
            return min(cands, key=lambda r: r["total_fuel"]), True, np.nan
    start = _fan_burner_grid(M, h, F_req)
    if start is None:
        return None, False, np.inf
    pi0, T40, r0 = start
    r = _slsqp(M, h, F_req, pi0, T40)
    if r is not None and r["total_fuel"] <= r0["total_fuel"]:
        return r, True, r0["total_fuel"]
    return r0, False, r0["total_fuel"]


def optimize_point(M, h, F_req=None, modes=MODES, x0=None):
    """
    Minimum-fuel operating point at (M, h).

    modes : subset of MODES to consider (baselines restrict this)
    x0    : optional (π_f, T4) warm start for the fan_burner branch

    Returns dict: M, h, F_req, mode (None if nothing feasible), pi_f, T4
    (None if burner off), P_gen, F, mdot_f_burner, mdot_f_gen, total_fuel,
    TSFC, refined, grid_fuel, branch_fuel {mode: fuel or inf},
    x_fan_burner ((π_f, T4) of the fan_burner solution, for warm starts).
    """
    if F_req is None:
        F_req = float(thrust_required(M, h))

    cand = dict.fromkeys(MODES)
    refined, grid_fuel = False, np.nan
    if "fan_burner" in modes:
        cand["fan_burner"], refined, grid_fuel = _fan_burner(M, h, F_req, x0)
    if "fan_only" in modes:
        cand["fan_only"] = _fan_only(M, h, F_req)
    if "ram" in modes:
        cand["ram"] = _ram(M, h, F_req)
    branch_fuel = {k: (float(r["total_fuel"]) if r is not None else np.inf)
                   for k, r in cand.items()}

    fb = cand["fan_burner"]
    out = {"M": M, "h": h, "F_req": F_req, "branch_fuel": branch_fuel,
           "refined": refined, "grid_fuel": grid_fuel,
           "x_fan_burner": None if fb is None
           else (float(fb["P25"] / fb["P02"]), float(fb["T4"]))}
    feasible = [k for k in modes if cand[k] is not None]
    if not feasible:
        return {**out, "mode": None, "pi_f": None, "T4": None, "P_gen": None,
                "F": None, "mdot_f_burner": None, "mdot_f_gen": None,
                "total_fuel": np.inf, "TSFC": np.inf}

    mode = min(feasible, key=lambda k: branch_fuel[k])
    r = cand[mode]
    return {
        **out,
        "mode": mode,
        "pi_f": float(r["P25"] / r["P02"]),
        "T4": float(r["T4"]) if r["burner_on"] else None,
        "P_gen": float(r["P_gen"]),
        "F": float(r["F"]),
        "mdot_f_burner": float(r["mdot_f"]),
        "mdot_f_gen": float(r["mdot_f_gen"]),
        "total_fuel": float(r["total_fuel"]),
        "TSFC": float(r["TSFC"]),
    }
