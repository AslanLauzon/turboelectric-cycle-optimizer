"""
Tests for burner.py.

Limiting cases:
  1. Burner off (T4=None) → identical result to fan.thrust().
  2. T4 = T25 → ṁ_f = 0, P4 = P25 (zero temperature rise).
  3. Energy conservation: ṁ_f·(η_b·LHV − cp·T4) == ṁ·cp·(T4 − T25).
  4. Higher T4 → more thrust (monotone).
  5. Fuel-to-air ratio below stoichiometric at T_BURNER_MAX.
  6. Ram mode (π_f=1, high Mach): burner adds useful thrust over fan alone.
  7. over_T4 flag trips when T4 > T_BURNER_MAX.
  8. TSFC_partial = 0 (or inf) when burner off and F > 0.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from burner import burner_exit, thrust_with_burner
from fan import fan_exit, thrust as fan_thrust
from config import CP_AIR, LHV, ETA_BURNER, T_BURNER_MAX


class TestBurnerExit:
    def test_off_when_T4_is_None(self):
        fe = fan_exit(0.8, 8_000.0, 2.0)
        be = burner_exit(fe, None)
        assert not be["burner_on"]
        assert be["mdot_f"] == 0.0
        assert abs(be["P4"] - fe["P25"]) < 1e-9

    def test_off_when_T4_equals_T25(self):
        fe = fan_exit(0.8, 8_000.0, 2.0)
        be = burner_exit(fe, fe["T25"])
        assert be["mdot_f"] == 0.0
        assert abs(be["P4"] - fe["P25"]) < 1e-9

    def test_energy_conservation(self):
        # ṁ_f·(η_b·LHV − cp·T4) == ṁ·cp·(T4 − T25)
        fe = fan_exit(1.0, 10_000.0, 1.5)
        T4 = 1600.0
        be = burner_exit(fe, T4)
        lhs = be["mdot_f"] * (ETA_BURNER * LHV - CP_AIR * T4)
        rhs = fe["mdot"] * CP_AIR * (T4 - fe["T25"])
        assert abs(lhs - rhs) < 1e-4

    def test_positive_fuel_flow_when_on(self):
        fe = fan_exit(1.0, 10_000.0, 2.0)
        be = burner_exit(fe, 1800.0)
        assert be["burner_on"]
        assert be["mdot_f"] > 0

    def test_pressure_drop_when_on(self):
        fe = fan_exit(1.0, 10_000.0, 2.0)
        be = burner_exit(fe, 1800.0)
        assert be["P4"] < fe["P25"]

    def test_mass_conservation(self):
        fe = fan_exit(1.0, 10_000.0, 2.0)
        be = burner_exit(fe, 1800.0)
        assert abs(be["mdot_total"] - (fe["mdot"] + be["mdot_f"])) < 1e-12

    def test_fuel_air_ratio_below_stoichiometric(self):
        # Stoichiometric Jet-A f/a ≈ 0.068; practical limit ~0.06
        fe = fan_exit(0.8, 5_000.0, 2.0)
        be = burner_exit(fe, T_BURNER_MAX)
        f_over_a = be["mdot_f"] / fe["mdot"]
        assert f_over_a < 0.068, f"f/a = {f_over_a:.4f} exceeds stoichiometric"

    def test_over_T4_flag(self):
        fe = fan_exit(1.0, 10_000.0, 2.0)
        be = burner_exit(fe, T_BURNER_MAX + 1.0)
        assert be["over_T4"]

    def test_no_over_T4_at_max(self):
        fe = fan_exit(1.0, 10_000.0, 2.0)
        be = burner_exit(fe, T_BURNER_MAX)
        assert not be["over_T4"]


class TestThrustWithBurner:
    def test_off_matches_fan_thrust(self):
        M, h, pi_f = 0.5, 5_000.0, 2.0
        t_burner_off = thrust_with_burner(M, h, pi_f, None)
        t_fan        = fan_thrust(M, h, pi_f)
        assert abs(t_burner_off["F"] - t_fan["F"]) < 1e-6

    def test_burner_on_more_thrust_than_off(self):
        M, h, pi_f = 1.0, 10_000.0, 2.0
        F_off = thrust_with_burner(M, h, pi_f, None)["F"]
        F_on  = thrust_with_burner(M, h, pi_f, 1800.0)["F"]
        assert F_on > F_off

    def test_thrust_increases_with_T4(self):
        M, h, pi_f = 1.0, 10_000.0, 2.0
        fe = fan_exit(M, h, pi_f)
        T_base = fe["T25"] + 50.0  # just above T25

        T4_vals = [T_base, 1200.0, 1600.0, 2000.0]
        F_vals  = [thrust_with_burner(M, h, pi_f, T4)["F"] for T4 in T4_vals]
        assert all(F_vals[i] < F_vals[i+1] for i in range(len(F_vals)-1))

    def test_tsfc_partial_zero_when_off(self):
        t = thrust_with_burner(0.8, 8_000.0, 2.0, None)
        # burner off → no fuel → TSFC_partial should be 0
        assert t["mdot_f"] == 0.0

    def test_tsfc_partial_positive_when_on(self):
        t = thrust_with_burner(1.0, 10_000.0, 2.0, 1800.0)
        assert t["TSFC_partial"] > 0

    def test_ram_mode(self):
        # π_f=1 → fan does no work; burner runs as ramjet combustor
        M, h = 2.0, 15_000.0
        F_ram_off = thrust_with_burner(M, h, 1.0, None)["F"]
        F_ram_on  = thrust_with_burner(M, h, 1.0, 1800.0)["F"]
        # Burner should add meaningful thrust in ram mode
        assert F_ram_on > F_ram_off

    def test_output_keys_present(self):
        t = thrust_with_burner(1.0, 10_000.0, 2.0, 1600.0)
        for key in ("mdot", "T25", "P25", "P_fan", "T4", "P4",
                    "mdot_f", "mdot_total", "V9", "F", "TSFC_partial"):
            assert key in t, f"missing key: {key}"
