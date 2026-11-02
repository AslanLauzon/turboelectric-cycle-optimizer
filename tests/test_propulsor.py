"""
Tests for propulsor.py — full propulsor assembly.

Key checks:
  1. Turbogen adds fuel cost but NOT thrust (F unchanged vs burner module alone).
  2. Full TSFC > TSFC_partial when P_gen > 0.
  3. Power balance constraint flags work correctly.
  4. Ram mode: pi_f=1 → P_fan=0 → P_gen=0 is feasible.
  5. TSFC increases with P_gen at fixed (M, h, pi_f, T4).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from propulsor import propulsor_point
from burner import thrust_with_burner
from fan import fan_exit
from config import ETA_CHAIN, P_GEN_MAX, P_FAN_MAX_CONT


class TestPowerSizing:
    def test_generator_can_drive_motors(self):
        # One turbogen must supply the full continuous motor rating
        assert ETA_CHAIN * P_GEN_MAX >= P_FAN_MAX_CONT

    def test_chain_roughly_balanced(self):
        # Generator not grossly oversized for one propulsor (within 10%)
        assert ETA_CHAIN * P_GEN_MAX < 1.1 * P_FAN_MAX_CONT


class TestThrustUnchanged:
    def test_turbogen_does_not_affect_thrust(self):
        M, h, pi_f, T4 = 0.8, 8_000.0, 2.0, None
        F_burner = thrust_with_burner(M, h, pi_f, T4)["F"]
        F_prop   = propulsor_point(M, h, pi_f, T4, 1.5e6)["F"]
        assert abs(F_prop - F_burner) < 1e-6

    def test_thrust_same_regardless_of_p_gen(self):
        M, h, pi_f, T4 = 1.0, 10_000.0, 2.0, 1600.0
        F_low  = propulsor_point(M, h, pi_f, T4, 0.5e6)["F"]
        F_high = propulsor_point(M, h, pi_f, T4, 3.0e6)["F"]
        assert abs(F_low - F_high) < 1e-6


class TestTSFC:
    def test_tsfc_includes_turbogen_fuel(self):
        M, h, pi_f, T4, P_gen = 0.8, 8_000.0, 2.0, None, 1.5e6
        TSFC_partial = thrust_with_burner(M, h, pi_f, T4)["TSFC_partial"]
        TSFC_full    = propulsor_point(M, h, pi_f, T4, P_gen)["TSFC"]
        assert TSFC_full > TSFC_partial

    def test_tsfc_increases_with_p_gen(self):
        # More generator power → more turbogen fuel → higher TSFC at same thrust
        M, h, pi_f, T4 = 0.8, 8_000.0, 2.0, None
        TSFC_low  = propulsor_point(M, h, pi_f, T4, 0.5e6)["TSFC"]
        TSFC_high = propulsor_point(M, h, pi_f, T4, 3.0e6)["TSFC"]
        assert TSFC_high > TSFC_low

    def test_tsfc_zero_fuel_when_all_off(self):
        # pi_f=1, no burner, no turbogen: no fuel at all (thrust may be 0 or small)
        result = propulsor_point(0.0, 0.0, 1.0, None, 0.0)
        assert result["total_fuel"] == 0.0


class TestPowerBalance:
    def test_feasible_when_p_gen_sufficient(self):
        M, h, pi_f = 0.5, 5_000.0, 2.0
        fe = fan_exit(M, h, pi_f)
        P_needed = fe["P_fan"] / ETA_CHAIN
        result = propulsor_point(M, h, pi_f, None, P_needed * 1.05)
        assert result["power_deficit"] == 0.0

    def test_infeasible_when_p_gen_too_small(self):
        M, h, pi_f = 0.5, 5_000.0, 2.0
        result = propulsor_point(M, h, pi_f, None, 0.0)
        fe = fan_exit(M, h, pi_f)
        if fe["P_fan"] > 0:
            assert result["power_deficit"] > 0

    def test_power_deficit_magnitude(self):
        M, h, pi_f = 0.5, 5_000.0, 2.0
        fe = fan_exit(M, h, pi_f)
        P_gen = fe["P_fan"] / ETA_CHAIN / 2.0   # half what's needed
        result = propulsor_point(M, h, pi_f, None, P_gen)
        expected_deficit = fe["P_fan"] - ETA_CHAIN * P_gen
        assert abs(result["power_deficit"] - expected_deficit) < 1e-4


class TestRamMode:
    def test_ram_mode_p_gen_zero_feasible(self):
        # pi_f=1 → P_fan=0 → no power needed → P_gen=0 is feasible
        result = propulsor_point(2.0, 15_000.0, 1.0, 1800.0, 0.0)
        assert result["power_deficit"] == 0.0

    def test_ram_mode_thrust_positive(self):
        result = propulsor_point(2.0, 15_000.0, 1.0, 1800.0, 0.0)
        assert result["F"] > 0

    def test_ram_mode_no_turbogen_fuel(self):
        result = propulsor_point(2.0, 15_000.0, 1.0, 1800.0, 0.0)
        assert result["mdot_f_gen"] == 0.0


class TestFeasibleFlag:
    def test_feasible_at_valid_point(self):
        M, h, pi_f = 0.5, 5_000.0, 1.5
        fe = fan_exit(M, h, pi_f)
        assert not fe["over_P"]
        P_gen = fe["P_fan"] / ETA_CHAIN * 1.1
        result = propulsor_point(M, h, pi_f, None, min(P_gen, P_GEN_MAX))
        assert result["feasible"]

    def test_infeasible_over_rated_turbogen(self):
        result = propulsor_point(0.5, 5_000.0, 1.5, None, P_GEN_MAX + 1.0)
        assert not result["feasible"]

    def test_output_keys_present(self):
        result = propulsor_point(1.0, 10_000.0, 2.0, 1600.0, 2.0e6)
        for key in ("F", "TSFC", "total_fuel", "power_deficit",
                    "P_available", "feasible", "mdot_f_gen", "V9"):
            assert key in result, f"missing key: {key}"
