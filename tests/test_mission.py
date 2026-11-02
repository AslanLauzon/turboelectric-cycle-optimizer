import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from mission import climb_altitude, drag_rise, dynamic_pressure, thrust_required
from diagnostics import calibrate_cda0, max_fan_thrust
from config import CLIMB_SCHEDULE, DRAG_RISE, F_REQ_FLOOR, M_FLOOR, CDA_0, GAMMA


class TestDragRise:
    def test_hits_breakpoints(self):
        for M, k in DRAG_RISE:
            assert abs(drag_rise(M) - k) < 1e-12

    def test_flat_subsonic(self):
        assert np.all(drag_rise(np.linspace(0.0, 0.8, 9)) == 1.0)


class TestThrustRequired:
    def test_dynamic_pressure(self):
        # q = ½ρV² = (γ/2)·P·M² at sea level, M = 1
        assert abs(dynamic_pressure(1.0, 0.0) - 0.5 * GAMMA * 101_325.0) < 1e-6

    def test_floor_below_m_floor(self):
        M = np.linspace(0.0, M_FLOOR - 1e-6, 20)
        assert np.all(thrust_required(M, 0.0) >= F_REQ_FLOOR)

    def test_no_floor_above_m_floor(self):
        M, h = 0.6, float(climb_altitude(0.6))
        expected = dynamic_pressure(M, h) * CDA_0 * drag_rise(M)
        assert abs(thrust_required(M, h) - expected) < 1e-9

    def test_linear_in_cda0(self):
        M, h = 1.5, 11_000.0
        assert abs(thrust_required(M, h, 2 * CDA_0) / thrust_required(M, h) - 2.0) < 1e-12


class TestCalibration:
    def test_config_matches_calibration(self):
        # Fails if the propulsor model changes without recalibrating CDA_0
        assert abs(CDA_0 / calibrate_cda0() - 1.0) < 1e-3

    def test_anchor_condition(self):
        M = 0.6
        h = float(climb_altitude(M))
        ratio = thrust_required(M, h) / max_fan_thrust(M, h)["F"]
        assert abs(ratio - 0.75) < 1e-3


class TestClimbAltitude:
    def test_hits_breakpoints(self):
        for M, h in CLIMB_SCHEDULE:
            assert abs(climb_altitude(M) - h) < 1e-9

    def test_monotone_non_decreasing(self):
        h = climb_altitude(np.linspace(0.0, 3.0, 301))
        assert np.all(np.diff(h) >= 0)

    def test_linear_between_breakpoints(self):
        # Midpoint of 0.8→1.5 segment: (3 km + 11 km)/2
        assert abs(climb_altitude(1.15) - 7_000.0) < 1e-9

    def test_vectorized(self):
        assert climb_altitude(np.array([0.0, 1.0, 3.0])).shape == (3,)
