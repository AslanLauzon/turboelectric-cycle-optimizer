"""
Tests for fan.py.

Limiting-case checks against Mattingly ideal turbofan (no burner, ideal nozzle):
  1. π_f = 1: no work done, T25=T02, P_fan=0.
  2. Fan-face sizing: corrected flow constant, ṁ > 0 at M = 0.
  3. Thrust increases monotonically with π_f.
  4. Energy conservation: P_fan = ṁ·cp·ΔT exactly.
  5. F = 0 at M = 0 with π_f = 1; F > 0 at M = 0 with π_f > 1.
  6. Power-limit flag trips correctly.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fan import mfp, mass_flow, capture_area, fan_exit, nozzle_velocity, thrust
from inlet import inlet_exit
from atmosphere import freestream
from config import CP_AIR, A_FAN, M_FACE


class TestMFP:
    def test_zero_at_zero_mach(self):
        assert mfp(0.0) == 0.0

    def test_choked_value(self):
        # γ=1.4, R=287.058: MFP(1) = √(γ/R)·(1.2)^-3 ≈ 0.040418 [Mattingly]
        assert abs(mfp(1.0) - 0.040418) < 1e-5

    def test_maximum_at_mach_one(self):
        M = np.linspace(0.0, 2.0, 2001)
        assert abs(M[np.argmax(mfp(M))] - 1.0) < 1e-3


class TestMassFlow:
    def test_sea_level_static_exact(self):
        # No ram rise or inlet loss: ṁ = MFP(M_face)·A·P_SL/√T_SL
        expected = mfp(M_FACE) * A_FAN * 101_325.0 / np.sqrt(288.15)
        assert abs(mass_flow(0.0, 0.0) - expected) < 1e-9

    def test_spot_values(self):
        # A_FAN = 0.5 m², M_FACE = 0.55
        assert 94.0 < mass_flow(0.0, 0.0) < 98.0
        assert 104.0 < mass_flow(3.0, 20_000.0) < 108.0

    def test_corrected_flow_constant(self):
        # ṁ·√θ2/δ2 is fixed by the fan face
        for M, h in [(0.0, 0.0), (0.8, 3_000.0), (2.0, 15_000.0), (3.0, 20_000.0)]:
            ie = inlet_exit(M, h)
            mc = mass_flow(M, h) * np.sqrt(ie["T02"] / 288.15) / (ie["P02"] / 101_325.0)
            assert abs(mc - mass_flow(0.0, 0.0)) < 1e-9

    def test_decreases_with_altitude_at_fixed_mach(self):
        assert mass_flow(1.0, 15_000.0) < mass_flow(1.0, 0.0)

    def test_vectorized(self):
        M = np.array([0.0, 1.0, 2.0])
        h = np.array([0.0, 10_000.0, 15_000.0])
        mdot = mass_flow(M, h)
        assert mdot.shape == (3,)
        assert np.all(mdot > 0)


class TestCaptureArea:
    def test_infinite_at_static(self):
        assert np.isinf(capture_area(0.0, 0.0))

    def test_definition(self):
        M, h = 2.0, 15_000.0
        fs = freestream(M, h)
        assert abs(capture_area(M, h) - mass_flow(M, h) / (fs["rho_s"] * fs["V"])) < 1e-12

    def test_reported_in_fan_exit(self):
        assert abs(fan_exit(2.0, 15_000.0, 1.0)["A0"] - capture_area(2.0, 15_000.0)) < 1e-12


class TestFanExit:
    def test_unity_pr_no_work(self):
        # π_f = 1 → T25 = T02, P_fan = 0
        fe = fan_exit(0.5, 5_000.0, 1.0)
        assert abs(fe["T25"] - fe["T02"]) < 1e-9
        assert abs(fe["P_fan"]) < 1e-6

    def test_unity_pr_no_pressure_rise(self):
        fe = fan_exit(0.5, 5_000.0, 1.0)
        assert abs(fe["P25"] - fe["P02"]) < 1e-6

    def test_energy_conservation(self):
        # P_fan == ṁ·cp·ΔT to floating-point precision
        fe = fan_exit(1.0, 10_000.0, 2.0)
        expected = fe["mdot"] * CP_AIR * (fe["T25"] - fe["T02"])
        assert abs(fe["P_fan"] - expected) < 1e-6

    def test_temperature_rises_with_pr(self):
        h, M = 10_000.0, 1.0
        T25_low  = fan_exit(M, h, 1.5)["T25"]
        T25_high = fan_exit(M, h, 3.0)["T25"]
        assert T25_high > T25_low

    def test_over_power_flag(self):
        # π_f=3.5 at M=0.3 SL is far above any motor limit
        fe = fan_exit(0.3, 0.0, 3.5)
        assert fe["over_P"]

    def test_no_over_power_at_low_pr(self):
        # π_f=1.05 needs only a small fraction of rated power
        fe = fan_exit(0.3, 0.0, 1.05)
        assert not fe["over_P"]


class TestNozzleVelocity:
    def test_zero_when_total_equals_static(self):
        # If P_t = P_s, no expansion → V9 = 0
        v = nozzle_velocity(300.0, 101325.0, 101325.0)
        assert abs(v) < 1e-9

    def test_positive_for_higher_total_pressure(self):
        v = nozzle_velocity(400.0, 200_000.0, 50_000.0)
        assert v > 0


class TestThrust:
    def test_zero_thrust_at_mach_zero(self):
        # M=0: V0=0, but also nozzle pressure ratio → V9 depends only on fan compression.
        # With π_f=1 (no work): V9=0 → F=0.
        t = thrust(0.0, 0.0, 1.0)
        assert abs(t["F"]) < 1e-3

    def test_static_thrust_positive_with_fan_work(self):
        assert thrust(0.0, 0.0, 1.3)["F"] > 0

    def test_tsfc_is_zero(self):
        # No burner in step 2: TSFC must be zero
        t = thrust(1.0, 10_000.0, 2.0)
        assert t["TSFC"] == 0.0

    def test_thrust_increases_with_pr(self):
        # At fixed (M, h), more PR → more nozzle velocity → more thrust
        h, M = 10_000.0, 0.8
        F_low  = thrust(M, h, 1.2)["F"]
        F_high = thrust(M, h, 2.5)["F"]
        assert F_high > F_low

    def test_thrust_positive_for_supersonic_fan(self):
        # At M=1.5, π_f=2: the fan adds enough energy to exceed ram drag
        t = thrust(1.5, 11_000.0, 2.0)
        assert t["F"] > 0

    def test_output_keys(self):
        t = thrust(0.8, 8_000.0, 2.0)
        for key in ("mdot", "T25", "P25", "P_fan", "V9", "F", "TSFC"):
            assert key in t
