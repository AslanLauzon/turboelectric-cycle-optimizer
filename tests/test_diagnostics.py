"""
Tests for diagnostics.max_fan_thrust — limiting cases only, no expected
thrust values.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import diagnostics
from diagnostics import max_fan_thrust, P_LIM
from propulsor import propulsor_point
from mission import climb_altitude
from config import T_FAN_EXIT_MAX, PI_F_MAX


def _on_schedule(M):
    return M, float(climb_altitude(M))


class TestPowerLimited:
    def test_power_binds_at_limit(self):
        d = max_fan_thrust(*_on_schedule(0.6))
        assert d["binding"] == "power"
        assert abs(d["P_fan"] / P_LIM - 1.0) < 1e-8

    def test_point_is_feasible(self):
        for M in (0.0, 0.8, 1.5, 3.0):
            M, h = _on_schedule(M)
            d = max_fan_thrust(M, h)
            assert propulsor_point(M, h, d["pi_f"], None, d["P_gen"])["feasible"]

    def test_limit_is_active(self):
        # 1% more π_f must violate a constraint
        M, h = _on_schedule(0.8)
        d = max_fan_thrust(M, h)
        r = propulsor_point(M, h, d["pi_f"] * 1.01, None, d["P_gen"])
        assert not r["feasible"]


class TestOtherLimits:
    def test_T25_binds_when_power_unlimited(self, monkeypatch):
        # M3 / 20 km: T02 ≈ 607 K, so T25 limit (π_f ≈ 3.37) is below PI_F_MAX
        monkeypatch.setattr(diagnostics, "P_LIM", 1e12)
        d = max_fan_thrust(*_on_schedule(3.0))
        assert d["binding"] == "T25"
        assert abs(d["T25"] - T_FAN_EXIT_MAX) < 1e-3

    def test_pi_f_max_binds_when_cool_and_unlimited(self, monkeypatch):
        monkeypatch.setattr(diagnostics, "P_LIM", 1e12)
        d = max_fan_thrust(0.0, 0.0)
        assert d["binding"] == "pi_f_max"
        assert abs(d["pi_f"] - PI_F_MAX) < 1e-6


class TestBurnerOff:
    def test_only_generator_fuel(self):
        M, h = _on_schedule(0.8)
        d = max_fan_thrust(M, h)
        r = propulsor_point(M, h, d["pi_f"], None, d["P_gen"])
        assert r["mdot_f"] == 0.0
        assert abs(r["total_fuel"] - r["mdot_f_gen"]) < 1e-15

    def test_static_thrust_positive(self):
        assert max_fan_thrust(0.0, 0.0)["F"] > 0
