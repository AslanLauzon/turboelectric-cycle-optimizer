import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from turbogen import turbogen_state
from config import PSFC_GEN, P_GEN_MAX


class TestTurbogenState:
    def test_off_no_fuel(self):
        tg = turbogen_state(0.0)
        assert tg["mdot_f_gen"] == 0.0
        assert not tg["over_P"]

    def test_fuel_proportional_to_power(self):
        P = 2.0e6
        tg = turbogen_state(P)
        assert abs(tg["mdot_f_gen"] - P * PSFC_GEN) < 1e-12

    def test_rated_power_no_flag(self):
        tg = turbogen_state(P_GEN_MAX)
        assert not tg["over_P"]

    def test_over_rated_power_flag(self):
        tg = turbogen_state(P_GEN_MAX + 1.0)
        assert tg["over_P"]

    def test_psfc_magnitude(self):
        # Guards the kg/J unit: typical turboshaft 0.25–0.35 kg/(kW·h)
        assert 0.25 < PSFC_GEN * 3.6e6 < 0.35

    def test_rated_fuel_flow(self):
        # 4 MW at 0.30 kg/(kW·h) → 1200 kg/h ≈ 0.333 kg/s
        assert abs(turbogen_state(P_GEN_MAX)["mdot_f_gen"] - 0.3333) < 1e-3

    def test_p_gen_passthrough(self):
        tg = turbogen_state(1.5e6)
        assert tg["P_gen"] == 1.5e6
