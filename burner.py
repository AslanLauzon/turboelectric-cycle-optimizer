"""
Burner module (stations 25 → 4 → 9).

The burner adds fuel to the fan-compressed airstream, raising total temperature
from T25 to T4.  Decision variable is T4; fuel flow is solved from the energy
balance.  Pressure loss applies only when the burner is lit.

Stations:  25 = fan exit,  4 = burner exit,  9 = nozzle exit

nozzle_velocity lives in fan.py and is shared with propulsor.py.
"""

import numpy as np
from atmosphere import freestream
from fan import fan_exit, nozzle_velocity
from config import (
    CP_AIR, LHV,
    ETA_BURNER, DELTA_P_BURNER, T_BURNER_MAX,
)


def burner_exit(fan_state, T4):
    """
    Burner-exit state given fan-exit state and desired exit temperature T4.

    Parameters
    ----------
    fan_state : dict
        Output of fan.fan_exit() — must contain keys: T25, P25, mdot.
    T4 : float or None
        Burner exit total temperature [K].  None or T4 <= T25 → burner off.

    Returns
    -------
    dict with keys:
        T4          burner exit total temperature [K]
        P4          burner exit total pressure [Pa]
        mdot_f      fuel mass flow [kg/s]  (0 if off)
        mdot_total  ṁ + ṁ_f [kg/s]
        burner_on   bool
        over_T4     bool — T4 exceeds T_BURNER_MAX
    """
    T25  = fan_state["T25"]
    P25  = fan_state["P25"]
    mdot = fan_state["mdot"]

    burner_on = (T4 is not None) and (T4 > T25)

    if burner_on:
        # Energy balance: ṁ·cp·T25 + ṁ_f·η_b·LHV = (ṁ + ṁ_f)·cp·T4
        mdot_f = mdot * CP_AIR * (T4 - T25) / (ETA_BURNER * LHV - CP_AIR * T4)
        P4     = P25 * (1.0 - DELTA_P_BURNER)
    else:
        T4     = T25   # normalise for downstream consistency
        mdot_f = 0.0
        P4     = P25

    return {
        "T4":         T4,
        "P4":         P4,
        "mdot_f":     mdot_f,
        "mdot_total": mdot + mdot_f,
        "burner_on":  burner_on,
        "over_T4":    T4 > T_BURNER_MAX,
    }


def thrust_with_burner(M, h, pi_f, T4):
    """
    Net thrust and fuel flow for fan + burner operation.

    Parameters
    ----------
    M, h   : flight condition
    pi_f   : fan pressure ratio (decision variable)
    T4     : burner exit temperature [K], or None for burner off

    Returns
    -------
    dict with all fan_exit keys, all burner_exit keys, plus:
        V9            nozzle exit velocity [m/s]
        F             net thrust [N]
        TSFC_partial  ṁ_f / F [kg/(N·s)] — turbogen fuel added in step 4
    """
    fe  = fan_exit(M, h, pi_f)
    be  = burner_exit(fe, T4)
    P_s = freestream(M, h)["P_s"]

    V9 = nozzle_velocity(be["T4"], be["P4"], P_s)
    F  = be["mdot_total"] * V9 - fe["mdot"] * fe["V0"]

    TSFC_partial = be["mdot_f"] / F if F > 0 else np.inf

    return {**fe, **be, "V9": V9, "F": F, "TSFC_partial": TSFC_partial}
