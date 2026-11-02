"""
Tests for optimizer.optimize_point — constraint satisfaction and optimality
checks only. Predicted mode-switch Mach numbers are results, not tested.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import itertools
import numpy as np
import pytest

from optimizer import optimize_point
from propulsor import propulsor_point
from mission import climb_altitude
from config import ETA_CHAIN, P_GEN_MAX, PI_F_MIN, PI_F_MAX, T_BURNER_MAX

MACHS = (0.2, 0.5, 0.6, 0.8, 1.05, 1.5, 2.5, 3.0)


@pytest.fixture(scope="module")
def results():
    return {M: optimize_point(M, float(climb_altitude(M))) for M in MACHS}


def _replay(r):
    return propulsor_point(r["M"], r["h"], r["pi_f"], r["T4"], r["P_gen"])


class TestConstraints:
    @pytest.mark.parametrize("M", MACHS)
    def test_feasible_and_meets_thrust(self, results, M):
        r = results[M]
        assert r["mode"] is not None
        p = _replay(r)
        assert p["feasible"]
        assert p["F"] >= r["F_req"] * (1.0 - 1e-6)

    @pytest.mark.parametrize("M", MACHS)
    def test_thrust_constraint_active(self, results, M):
        # Minimum fuel should not over-deliver thrust
        r = results[M]
        assert r["F"] / r["F_req"] - 1.0 < 1e-3

    @pytest.mark.parametrize("M", MACHS)
    def test_generator_sized_to_fan(self, results, M):
        r = results[M]
        P_fan = float(_replay(r)["P_fan"])
        if r["mode"] == "ram":
            assert r["P_gen"] == 0.0
        else:
            assert abs(r["P_gen"] * ETA_CHAIN / P_fan - 1.0) < 1e-9


class TestModeSelection:
    def test_below_crossover_no_burner_fuel(self, results):
        assert results[0.5]["mdot_f_burner"] == 0.0

    def test_above_crossover_burner_lit(self, results):
        assert results[0.8]["mdot_f_burner"] > 0.0

    @pytest.mark.parametrize("M", MACHS)
    def test_picks_cheapest_branch(self, results, M):
        r = results[M]
        assert r["total_fuel"] == min(r["branch_fuel"].values())

    @pytest.mark.parametrize("M", MACHS)
    def test_slsqp_not_worse_than_grid(self, results, M):
        r = results[M]
        assert r["branch_fuel"]["fan_burner"] <= r["grid_fuel"]


class TestBruteForce:
    def test_3d_grid_finds_nothing_cheaper(self, results):
        # Coarse grid over (π_f, T4, P_gen) incl. oversized P_gen, burner off and fan off
        M = 1.5
        h = float(climb_altitude(M))
        r = results[M]
        best = np.inf
        pis = np.concatenate(([1.0], np.linspace(PI_F_MIN, PI_F_MAX, 12)))
        T4s = [None] + list(np.linspace(400.0, T_BURNER_MAX, 12))
        for pi, T4 in itertools.product(pis, T4s):
            P_fan = float(propulsor_point(M, h, pi, None, 0.0)["P_fan"])
            for k in (1.0, 1.2, 1.5):
                P_gen = min(P_fan / ETA_CHAIN * k * (1 + 1e-12), P_GEN_MAX)
                p = propulsor_point(M, h, pi, T4, P_gen)
                if p["feasible"] and p["F"] >= r["F_req"]:
                    best = min(best, float(p["total_fuel"]))
        assert np.isfinite(best)
        assert r["total_fuel"] <= best * (1.0 + 1e-6)


class TestInfeasible:
    def test_impossible_requirement(self):
        r = optimize_point(1.5, 11_000.0, F_req=1e6)
        assert r["mode"] is None
        assert all(np.isinf(v) for v in r["branch_fuel"].values())
