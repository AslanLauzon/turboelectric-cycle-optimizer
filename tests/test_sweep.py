"""
Tests for sweep.py — warm starts, baselines, transition finding.
Baselines are restricted versions of the optimizer, so it must never lose.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from sweep import (mach_sweep, find_transitions, power_down_onset,
                   fixed_pi_design, fixed_pi_point, _on_schedule, TRANSITION_TOL)
from optimizer import optimize_point
from diagnostics import max_fan_thrust, P_LIM
from propulsor import propulsor_point
from config import ETA_CHAIN

M_TEST = np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8])


@pytest.fixture(scope="module")
def opt():
    return mach_sweep(M_TEST)


@pytest.fixture(scope="module")
def pi_fix():
    return fixed_pi_design()


class TestWarmStart:
    @pytest.mark.parametrize("i", [2, 4, 5])
    def test_warm_matches_cold(self, opt, i):
        cold = optimize_point(*_on_schedule(M_TEST[i]))
        assert abs(opt[i]["total_fuel"] / cold["total_fuel"] - 1.0) < 1e-6
        assert opt[i]["mode"] == cold["mode"]


class TestBaselines:
    def test_optimizer_beats_turbojet(self, opt):
        tj = mach_sweep(M_TEST, modes=("fan_burner",))
        for r, b in zip(opt, tj):
            assert r["total_fuel"] <= b["total_fuel"] * (1.0 + 1e-6)

    def test_turbojet_always_burning(self):
        for r in mach_sweep(M_TEST[:3], modes=("fan_burner",)):
            assert r["mode"] == "fan_burner" and r["mdot_f_burner"] > 0

    def test_optimizer_beats_fixed_pi(self, opt, pi_fix):
        for r in opt:
            b = fixed_pi_point(r["M"], r["h"], pi_fix)
            assert r["total_fuel"] <= b["total_fuel"] * (1.0 + 1e-6)

    def test_fixed_pi_feasible_everywhere(self, pi_fix):
        for m in M_TEST:
            M, h = _on_schedule(m)
            b = fixed_pi_point(M, h, pi_fix)
            assert np.isfinite(b["total_fuel"])
            p = propulsor_point(M, h, pi_fix, b["T4"], b["P_gen"])
            assert p["feasible"] and p["F"] >= b["F_req"] * (1.0 - 1e-6)

    def test_fixed_pi_design_is_binding_minimum(self, pi_fix):
        grid = np.linspace(0.0, 3.0, 121)
        limits = [max_fan_thrust(*_on_schedule(m))["pi_f"] for m in grid]
        assert all(pi_fix <= p for p in limits)
        assert abs(pi_fix - min(limits)) < 1e-12


class TestTransitions:
    def test_bisected_switch_brackets_mode_change(self, opt):
        for m, before, after in find_transitions(opt):
            lo = optimize_point(*_on_schedule(m - TRANSITION_TOL))
            hi = optimize_point(*_on_schedule(m + TRANSITION_TOL))
            assert lo["mode"] == before
            assert hi["mode"] == after

    def test_power_down_onset_synthetic(self):
        P = P_LIM / ETA_CHAIN
        rows = [
            {"M": 0.5, "mode": "fan_only", "P_gen": 0.5 * P},
            {"M": 1.0, "mode": "fan_burner", "P_gen": P},
            {"M": 1.5, "mode": "fan_burner", "P_gen": P},
            {"M": 1.8, "mode": "fan_burner", "P_gen": 0.8 * P},
            {"M": 2.2, "mode": "ram", "P_gen": 0.0},
        ]
        assert power_down_onset(rows) == 1.8

    def test_power_down_onset_straight_to_ram(self):
        P = P_LIM / ETA_CHAIN
        rows = [
            {"M": 1.0, "mode": "fan_burner", "P_gen": P},
            {"M": 2.0, "mode": "ram", "P_gen": 0.0},
        ]
        assert power_down_onset(rows) == 2.0
