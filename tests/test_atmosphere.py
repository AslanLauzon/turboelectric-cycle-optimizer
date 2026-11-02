"""
Tests for atmosphere.py — checked against ICAO Doc 7488 / US Standard Atmosphere 1976.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atmosphere import temperature, pressure, density, speed_of_sound, freestream, T_SL, P_SL

ABS = 1e-3  # tight absolute tolerance for known tabulated values


class TestTemperature:
    def test_sea_level(self):
        assert abs(temperature(0) - 288.15) < ABS

    def test_tropopause(self):
        # ISA: T(11000) = 216.65 K exactly
        assert abs(temperature(11_000) - 216.65) < ABS

    def test_isothermal_stratosphere(self):
        # 11–20 km is isothermal at 216.65 K
        assert abs(temperature(15_000) - 216.65) < ABS
        assert abs(temperature(20_000) - 216.65) < ABS

    def test_upper_stratosphere(self):
        # 20–32 km warms at +1 K/km
        assert abs(temperature(25_000) - (216.65 + 1.0 * 5.0)) < ABS

    def test_vectorized(self):
        h = np.array([0.0, 5_000.0, 11_000.0, 15_000.0])
        T = temperature(h)
        assert T.shape == (4,)
        assert abs(T[0] - 288.15) < ABS


class TestPressure:
    def test_sea_level(self):
        assert abs(pressure(0) - 101_325.0) < 0.1  # Pa

    def test_tropopause(self):
        # ICAO tabulated: 22632.1 Pa at 11 km
        assert abs(pressure(11_000) - 22_632.1) < 2.0

    def test_stratosphere_20km(self):
        # ICAO tabulated: 5474.9 Pa at 20 km
        assert abs(pressure(20_000) - 5_474.9) < 2.0

    def test_monotone_decreasing(self):
        h = np.linspace(0, 30_000, 100)
        p = pressure(h)
        assert np.all(np.diff(p) < 0)


class TestDensity:
    def test_sea_level(self):
        # ISA: 1.225 kg/m³
        assert abs(density(0) - 1.225) < 1e-3

    def test_ideal_gas(self):
        from config import R_AIR
        h = np.array([0.0, 5_000.0, 15_000.0])
        assert np.allclose(density(h), pressure(h) / (R_AIR * temperature(h)))


class TestSpeedOfSound:
    def test_sea_level(self):
        # a_SL = sqrt(1.4 * 287.058 * 288.15) ≈ 340.29 m/s
        assert abs(speed_of_sound(0) - 340.29) < 0.1

    def test_tropopause(self):
        # T=216.65 K → a ≈ 295.07 m/s
        assert abs(speed_of_sound(11_000) - 295.07) < 0.1


class TestFreestream:
    def test_subsonic_stagnation(self):
        # M=0.3, sea level: T0 should be slightly above T_s
        fs = freestream(0.3, 0.0)
        assert fs["T0"] > fs["T_s"]
        assert fs["P0"] > fs["P_s"]

    def test_static_at_mach_zero(self):
        fs = freestream(0.0, 0.0)
        assert abs(fs["T0"] - fs["T_s"]) < ABS
        assert abs(fs["P0"] - fs["P_s"]) < 0.1

    def test_velocity(self):
        # M=1 at sea level: V should equal speed of sound
        fs = freestream(1.0, 0.0)
        assert abs(fs["V"] - speed_of_sound(0.0)) < 0.1

    def test_total_temperature_M2_SL(self):
        # T0 = T * (1 + 0.2 * 4) = T * 1.8
        fs = freestream(2.0, 0.0)
        assert abs(fs["T0"] - T_SL * 1.8) < ABS
