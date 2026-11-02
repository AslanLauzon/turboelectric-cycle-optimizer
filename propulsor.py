"""
Full propulsor assembly (Step 4+).

Single entry point for the optimizer: chains inlet → fan → turbogen power
balance → burner → nozzle → thrust and reports all constraint violations.

    propulsor_point(M, h, pi_f, T4, P_gen) → dict

Decision variables
------------------
pi_f  : fan pressure ratio
T4    : burner exit temperature [K], or None for burner off
P_gen : turbogenerator electrical output [W]  (0 = turbogen off / ram mode)

Constraint flags (optimizer uses these to reject infeasible points)
-----------------------------------------------------------------
power_deficit   max(0, P_fan − η_chain·P_gen)  [W]  must be 0
over_T4         T4 > T_BURNER_MAX
over_P (fan)    P_fan > P_FAN_MAX_CONT
over_P (gen)    P_gen > P_GEN_MAX
feasible        True iff all flags clear

TSFC
----
Total TSFC = (ṁ_f,burner + ṁ_f,gen) / F  [kg/(N·s)]
Replaces the partial TSFC from burner.py which excluded turbogen fuel.
"""

import numpy as np
from atmosphere import freestream
from fan import fan_exit, nozzle_velocity
from burner import burner_exit
from turbogen import turbogen_state
from config import ETA_CHAIN


def propulsor_point(M, h, pi_f, T4, P_gen):
    """
    Evaluate the full propulsor at one operating point.

    Parameters
    ----------
    M, h   : float — Mach number and altitude [m]
    pi_f   : float — fan pressure ratio
    T4     : float or None — burner exit temperature [K]; None = burner off
    P_gen  : float — turbogenerator electrical output [W]; 0 = off

    Returns
    -------
    dict merging all sub-module outputs plus:
        V9            nozzle exit velocity [m/s]
        F             net thrust [N]
        total_fuel    ṁ_f,burner + ṁ_f,gen [kg/s]
        TSFC          total_fuel / F [kg/(N·s)]
        P_available   η_chain · P_gen [W]
        power_deficit max(0, P_fan − P_available) [W]
        feasible      bool
    """
    fe  = fan_exit(M, h, pi_f)
    tg  = turbogen_state(P_gen)
    be  = burner_exit(fe, T4)
    P_s = freestream(M, h)["P_s"]

    V9 = nozzle_velocity(be["T4"], be["P4"], P_s)
    F  = be["mdot_total"] * V9 - fe["mdot"] * fe["V0"]

    total_fuel    = be["mdot_f"] + tg["mdot_f_gen"]
    TSFC          = total_fuel / F if F > 0 else np.inf

    P_available   = ETA_CHAIN * P_gen
    power_deficit = max(0.0, float(fe["P_fan"]) - P_available)

    feasible = (
        power_deficit == 0.0
        and not bool(fe["over_P"])
        and not bool(fe["over_T"])
        and not bool(be["over_T4"])
        and not bool(tg["over_P"])
    )

    return {
        **fe, **be, **tg,
        "V9":           V9,
        "F":            F,
        "total_fuel":   total_fuel,
        "TSFC":         TSFC,
        "P_available":  P_available,
        "power_deficit": power_deficit,
        "feasible":     feasible,
    }
