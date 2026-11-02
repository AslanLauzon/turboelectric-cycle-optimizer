"""
Inlet module — total-pressure recovery using MIL-E-5007D.

The MIL curve is an empirical design-point estimate for a well-engineered
supersonic inlet (oblique-shock diffuser).  It is substantially better than
a normal-shock inlet (e.g. M=2: MIL gives 0.925 vs. normal-shock ~0.721)
and is the right reference for a serious design-point study.

The inlet is modeled as adiabatic: T02 = T00.  No shaft work, no heat transfer.

Stations:  0 = freestream,  2 = inlet exit (fan face)
"""

import numpy as np
from atmosphere import freestream


def total_pressure_recovery(M):
    """
    MIL-E-5007D inlet total-pressure recovery η_r(M).

    η_r = P02_total / P00_total
    Vectorized; accepts scalars or arrays.
    """
    M = np.asarray(M, dtype=float)
    return np.where(
        M <= 1.0,
        1.0,
        np.where(
            M <= 5.0,
            1.0 - 0.075 * np.maximum(M - 1.0, 0.0) ** 1.35,
            800.0 / (M**4 + 935.0),
        ),
    )


def inlet_exit(M, h):
    """
    Return inlet-exit total conditions (station 2) given flight (M, h).

    Returns
    -------
    dict with keys:
        T02   total temperature at inlet exit [K]  — equals T00 (adiabatic)
        P02   total pressure at inlet exit [Pa]
        eta_r inlet total-pressure recovery [-]
        T00   freestream total temperature [K]
        P00   freestream total pressure [Pa]
        V     freestream velocity [m/s]
    """
    fs = freestream(M, h)
    eta_r = total_pressure_recovery(M)

    return {
        "T02":   fs["T0"],
        "P02":   eta_r * fs["P0"],
        "eta_r": eta_r,
        "T00":   fs["T0"],
        "P00":   fs["P0"],
        "V":     fs["V"],
    }
