"""
Tests for inlet.py.

Key limiting-case checks:
  1. Below M 1.0: η_r = 1.0 (no shocks, ideal recovery).
  2. MIL-E-5007 at M=2 is much better than a normal-shock inlet
     → confirms the curve models an oblique-shock diffuser, not a pitot intake.
  3. Adiabatic inlet: T02 = T00.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from inlet import total_pressure_recovery, inlet_exit
from config import GAMMA


def normal_shock_pt_ratio(M):
    """
    Normal-shock total-pressure ratio Pt2/Pt1 at Mach M > 1.

    Rayleigh Pitot formula (Mattingly eq. 3.25):
        Pt2/Pt1 = [(g+1)M²/2 / (1 + (g-1)/2·M²)]^(g/(g-1))
                  × [(g+1) / (2gM² - (g-1))]^(1/(g-1))

    Returns values in (0, 1] — always a loss for M > 1.
    Verified: M=2 → 0.7209, M=1.5 → 0.9298.
    """
    g = GAMMA
    term_A = ((g + 1) * M**2 / 2.0 / (1.0 + (g - 1) / 2.0 * M**2)) ** (g / (g - 1))
    term_B = ((g + 1) / (2.0 * g * M**2 - (g - 1))) ** (1.0 / (g - 1))
    return term_A * term_B


class TestRecovery:
    def test_subsonic_perfect(self):
        for M in [0.0, 0.5, 0.99, 1.0]:
            assert abs(total_pressure_recovery(M) - 1.0) < 1e-10

    def test_M2_value(self):
        # 1 - 0.075 * (2-1)^1.35 = 0.925
        assert abs(total_pressure_recovery(2.0) - 0.925) < 1e-10

    def test_M3_value(self):
        # 1 - 0.075 * (3-1)^1.35 ≈ 1 - 0.075 * 2.5475 ≈ 0.8093
        expected = 1.0 - 0.075 * 2.0**1.35
        assert abs(total_pressure_recovery(3.0) - expected) < 1e-10

    def test_better_than_normal_shock_supersonic(self):
        # MIL curve assumes oblique-shock inlet — must beat a plain normal shock
        for M in [1.5, 2.0, 2.5, 3.0]:
            eta_mil = total_pressure_recovery(M)
            eta_ns  = normal_shock_pt_ratio(M)
            assert eta_mil > eta_ns, (
                f"M={M}: MIL η_r={eta_mil:.4f} should exceed normal-shock {eta_ns:.4f}"
            )

    def test_monotone_decreasing_supersonic(self):
        M = np.linspace(1.0, 5.0, 50)
        eta = total_pressure_recovery(M)
        assert np.all(np.diff(eta) <= 0)

    def test_vectorized(self):
        M = np.array([0.5, 1.0, 2.0, 3.0])
        eta = total_pressure_recovery(M)
        assert eta.shape == (4,)


class TestInletExit:
    def test_adiabatic(self):
        # T02 must equal T00 regardless of Mach (adiabatic inlet)
        for M in [0.3, 1.0, 2.0, 3.0]:
            ie = inlet_exit(M, 10_000.0)
            assert abs(ie["T02"] - ie["T00"]) < 1e-9

    def test_subsonic_no_pressure_loss(self):
        # Below M=1: η_r=1 → P02 = P00
        ie = inlet_exit(0.5, 0.0)
        assert abs(ie["P02"] - ie["P00"]) < 1e-6

    def test_supersonic_pressure_loss(self):
        # Above M=1: P02 < P00
        ie = inlet_exit(2.0, 10_000.0)
        assert ie["P02"] < ie["P00"]
        assert abs(ie["eta_r"] - 0.925) < 1e-10

    def test_total_temperature_rises_with_mach(self):
        # T00 should increase with M at fixed altitude (kinetic heating)
        ie_low  = inlet_exit(0.3, 10_000.0)
        ie_high = inlet_exit(2.0, 10_000.0)
        assert ie_high["T00"] > ie_low["T00"]
