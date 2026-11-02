"""
Mission profile: climb schedule h(M) and thrust requirement F_req(M, h).

    F_req = q · CDA_0 · k(M),   q = (γ/2)·P_s·M²
    F_req ≥ F_REQ_FLOOR for M < M_FLOOR

All breakpoint tables live in config.py.
"""

import numpy as np
from atmosphere import pressure
from config import (
    GAMMA, CLIMB_SCHEDULE, DRAG_RISE, F_REQ_FLOOR, M_FLOOR, CDA_0,
)

_M_PTS = np.array([p[0] for p in CLIMB_SCHEDULE])
_H_PTS = np.array([p[1] for p in CLIMB_SCHEDULE])
_M_K   = np.array([p[0] for p in DRAG_RISE])
_K     = np.array([p[1] for p in DRAG_RISE])


def climb_altitude(M):
    """Altitude [m] on the climb schedule at Mach M."""
    return np.interp(M, _M_PTS, _H_PTS)


def drag_rise(M):
    """Drag-rise factor k(M) [-]."""
    return np.interp(M, _M_K, _K)


def dynamic_pressure(M, h):
    """q = (γ/2)·P_s·M² [Pa]."""
    return 0.5 * GAMMA * pressure(h) * np.asarray(M, dtype=float) ** 2


def thrust_required(M, h, cda0=CDA_0):
    """Required net thrust [N] at (M, h)."""
    F = dynamic_pressure(M, h) * cda0 * drag_rise(M)
    return np.where(np.asarray(M) < M_FLOOR, np.maximum(F, F_REQ_FLOOR), F)
