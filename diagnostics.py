"""
Diagnostics: maximum fan-only thrust (burner off).

With an ideal nozzle, F rises monotonically with π_f, so the maximum is at the
largest allowable π_f. Each limit has a closed-form bound:

    power   P_fan = ṁ·cp·T02·(π_f^e − 1) ≤ P_lim
    T25     T02·π_f^e ≤ T_FAN_EXIT_MAX
    PR      π_f ≤ PI_F_MAX

with e = (γ−1)/(γ·η_p) and P_lim = min(P_FAN_MAX_CONT, η_chain·P_GEN_MAX).
The point is then evaluated through propulsor_point, the optimizer's path.
"""

from fan import fan_exit, _FAN_EXP
from propulsor import propulsor_point
from mission import climb_altitude, dynamic_pressure, drag_rise
from config import (
    CP_AIR, ETA_CHAIN, P_FAN_MAX_CONT, P_GEN_MAX,
    T_FAN_EXIT_MAX, PI_F_MAX,
)

P_LIM = min(P_FAN_MAX_CONT, ETA_CHAIN * P_GEN_MAX)


def max_fan_thrust(M, h):
    """
    Maximum fan-only thrust at (M, h). Scalar inputs.

    Returns dict: M, h, pi_f, F, P_fan, P_gen, TSFC, mdot, T25, binding,
    where binding is "power", "T25" or "pi_f_max".
    """
    fe = fan_exit(M, h, 1.0)
    mdot, T02 = float(fe["mdot"]), float(fe["T02"])

    bounds = {
        "power":    (1.0 + P_LIM / (mdot * CP_AIR * T02)) ** (1.0 / _FAN_EXP),
        "T25":      (T_FAN_EXIT_MAX / T02) ** (1.0 / _FAN_EXP),
        "pi_f_max": PI_F_MAX,
    }
    binding = min(bounds, key=bounds.get)
    pi_f = bounds[binding] * (1.0 - 1e-9)   # stay inside the limit despite round-off

    P_fan = float(fan_exit(M, h, pi_f)["P_fan"])
    P_gen = P_fan / ETA_CHAIN
    r = propulsor_point(M, h, pi_f, None, P_gen)

    return {
        "M": M, "h": h, "pi_f": pi_f,
        "F": float(r["F"]), "P_fan": P_fan, "P_gen": P_gen,
        "TSFC": float(r["TSFC"]), "mdot": mdot, "T25": float(r["T25"]),
        "binding": binding,
    }


def calibrate_cda0(M_cal=0.6, fraction=0.75):
    """CDA_0 [m²] such that F_req = fraction · F_max(fan-only) at M_cal on the climb schedule."""
    h = float(climb_altitude(M_cal))
    F_target = fraction * max_fan_thrust(M_cal, h)["F"]
    return F_target / float(dynamic_pressure(M_cal, h) * drag_rise(M_cal))
