"""
Fan-only propulsor module (stations 2 → 25 → 9, no burner).

Mass flow is set by the fan face (fixed corrected flow), not a decision
variable:

    ṁ = MFP(M_FACE) · A_FAN · P02 / √T02
    MFP(M) = √(γ/R) · M · (1 + (γ−1)/2·M²)^(−(γ+1)/(2(γ−1)))

Required capture area A0 = ṁ/(ρ0·V0) is a diagnostic output.

Decision variable: π_f (fan pressure ratio).

The nozzle is ideally expanded: P9 = P_s.

All functions vectorized over (M, h, pi_f).
"""

import numpy as np
from atmosphere import freestream
from inlet import inlet_exit
from config import (
    GAMMA, R_AIR, CP_AIR, A_FAN, M_FACE,
    ETA_FAN_POLY, P_FAN_MAX_CONT, T_FAN_EXIT_MAX,
)

# Polytropic exponent for a compressor stage: (γ-1)/(γ·η_p)
_FAN_EXP = (GAMMA - 1.0) / (GAMMA * ETA_FAN_POLY)


def mfp(M):
    """Mass-flow parameter ṁ√Tt/(A·Pt) [√K·s/m]."""
    M = np.asarray(M, dtype=float)
    return (np.sqrt(GAMMA / R_AIR) * M
            * (1.0 + 0.5 * (GAMMA - 1.0) * M**2)
            ** (-(GAMMA + 1.0) / (2.0 * (GAMMA - 1.0))))


def mass_flow(M, h):
    """Fan-face mass flow [kg/s]: MFP(M_FACE)·A_FAN·P02/√T02."""
    ie = inlet_exit(M, h)
    return mfp(M_FACE) * A_FAN * ie["P02"] / np.sqrt(ie["T02"])


def capture_area(M, h):
    """Required capture area A0 = ṁ/(ρ0·V0) [m²]; inf at M = 0."""
    fs = freestream(M, h)
    rho_V = np.asarray(fs["rho_s"] * fs["V"], dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(rho_V > 0, mass_flow(M, h) / rho_V, np.inf)


def fan_exit(M, h, pi_f):
    """
    Fan-exit thermodynamic state given flight (M, h) and pressure ratio pi_f.

    Returns
    -------
    dict with keys:
        mdot    fan-face mass flow [kg/s]
        A0      required capture area ṁ/(ρ0·V0) [m²] (diagnostic; inf at M=0)
        T25     fan-exit total temperature [K]
        P25     fan-exit total pressure [Pa]
        P_fan   shaft power required [W]
        T02     inlet-exit total temperature [K]  (= T25 when pi_f=1)
        P02     inlet-exit total pressure [Pa]
        V0      freestream velocity [m/s]
        over_T  bool — T25 exceeds T_FAN_EXIT_MAX
        over_P  bool — P_fan exceeds P_FAN_MAX_CONT
    """
    ie    = inlet_exit(M, h)
    T02   = ie["T02"]
    P02   = ie["P02"]
    mdot  = mfp(M_FACE) * A_FAN * P02 / np.sqrt(T02)

    T25   = T02 * pi_f ** _FAN_EXP
    P25   = P02 * pi_f
    P_fan = mdot * CP_AIR * (T25 - T02)

    return {
        "mdot":   mdot,
        "A0":     capture_area(M, h),
        "T25":    T25,
        "P25":    P25,
        "P_fan":  P_fan,
        "T02":    T02,
        "P02":    P02,
        "V0":     ie["V"],
        "over_T": T25 > T_FAN_EXIT_MAX,
        "over_P": P_fan > P_FAN_MAX_CONT,
    }


def nozzle_velocity(T_t, P_t, P_s):
    """
    Ideal nozzle exit velocity [m/s] for isentropic expansion
    from total state (T_t, P_t) to ambient static pressure P_s.
    """
    T9_s = T_t * (P_s / P_t) ** ((GAMMA - 1.0) / GAMMA)
    return np.sqrt(np.maximum(2.0 * CP_AIR * (T_t - T9_s), 0.0))


def thrust(M, h, pi_f):
    """
    Net thrust [N] for fan-only operation (no burner, no fuel mass correction).

    F = ṁ · (V9 − V0)

    Returns
    -------
    dict with all fan_exit keys plus:
        V9      nozzle-exit velocity [m/s]
        F       net thrust [N]
        TSFC    0.0 (no fuel burned in this step)
    """
    fe  = fan_exit(M, h, pi_f)
    P_s = freestream(M, h)["P_s"]
    V9  = nozzle_velocity(fe["T25"], fe["P25"], P_s)
    F   = fe["mdot"] * (V9 - fe["V0"])

    return {**fe, "V9": V9, "F": F, "TSFC": 0.0}
