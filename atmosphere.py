"""
ISA standard atmosphere, three layers (0–32 km).

Covers the Mach 0–3 flight envelope; Mach 3 cruise sits around 15–20 km,
so we need at minimum the isothermal stratosphere.

All functions are vectorized (accept numpy arrays).
"""

import numpy as np
from config import GAMMA, R_AIR, G

# Layer boundaries and lapse rates
_H_TROP    = 11_000.0  # m, tropopause
_H_STRAT1  = 20_000.0  # m, top of isothermal stratosphere
_L_TROP    = -6.5e-3   # K/m, tropospheric lapse rate
_L_STRAT1  = 0.0       # K/m, lower stratosphere (isothermal)
_L_STRAT2  = +1.0e-3   # K/m, upper stratosphere (20–32 km, warming)

T_SL = 288.15    # K, sea-level temperature
P_SL = 101325.0  # Pa, sea-level pressure

# Layer base temperatures (derived once)
_T_TROP  = T_SL + _L_TROP * _H_TROP        # 216.65 K
_T_STRAT = _T_TROP                           # same (isothermal layer above)

# Layer base pressures (derived once)
_P_TROP  = P_SL * (_T_TROP / T_SL) ** (-G / (_L_TROP * R_AIR))
_P_STRAT = _P_TROP * np.exp(-G * (_H_STRAT1 - _H_TROP) / (R_AIR * _T_TROP))


def temperature(h):
    """ISA temperature [K] at geometric altitude h [m]."""
    h = np.asarray(h, dtype=float)
    return np.where(
        h <= _H_TROP,
        T_SL + _L_TROP * h,
        np.where(
            h <= _H_STRAT1,
            _T_TROP,
            _T_TROP + _L_STRAT2 * (h - _H_STRAT1),
        ),
    )


def pressure(h):
    """ISA pressure [Pa] at geometric altitude h [m]."""
    h = np.asarray(h, dtype=float)
    T = temperature(h)
    return np.where(
        h <= _H_TROP,
        P_SL * (T / T_SL) ** (-G / (_L_TROP * R_AIR)),
        np.where(
            h <= _H_STRAT1,
            _P_TROP * np.exp(-G * (h - _H_TROP) / (R_AIR * _T_TROP)),
            _P_STRAT * (T / _T_STRAT) ** (-G / (_L_STRAT2 * R_AIR)),
        ),
    )


def density(h):
    """ISA density [kg/m³]."""
    return pressure(h) / (R_AIR * temperature(h))


def speed_of_sound(h):
    """ISA speed of sound [m/s]."""
    return np.sqrt(GAMMA * R_AIR * temperature(h))


def freestream(M, h):
    """
    Return a dict of freestream and stagnation quantities at (M, h).

    Keys: T_s, P_s, rho_s, a_s, V, T0, P0
    All scalars or arrays matching the shape of M and h.
    """
    T_s   = temperature(h)
    P_s   = pressure(h)
    rho_s = density(h)
    a_s   = speed_of_sound(h)
    V     = M * a_s

    tau   = 1.0 + 0.5 * (GAMMA - 1.0) * M**2   # T0/T
    delta = tau ** (GAMMA / (GAMMA - 1.0))       # P0/P  (isentropic)

    return {
        "T_s":   T_s,
        "P_s":   P_s,
        "rho_s": rho_s,
        "a_s":   a_s,
        "V":     V,
        "T0":    T_s * tau,
        "P0":    P_s * delta,
    }
