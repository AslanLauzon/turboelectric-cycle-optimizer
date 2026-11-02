"""
Turbogenerator model.

The turboshaft is thermodynamically decoupled from the propulsor — it has its
own air intake and exhaust.  The model is a black box: electrical output P_gen
in, fuel flow out.

    ṁ_f,gen = P_gen × PSFC_gen

PSFC_gen is constant (assumed); flagged for sensitivity analysis in step 7.
"""

from config import PSFC_GEN, P_GEN_MAX


def turbogen_state(P_gen):
    """
    Turbogenerator fuel flow for a given electrical output P_gen [W].

    Returns
    -------
    dict with keys:
        P_gen       electrical output [W]
        mdot_f_gen  fuel mass flow [kg/s]
        over_P      bool — P_gen exceeds rated maximum
    """
    return {
        "P_gen":      P_gen,
        "mdot_f_gen": P_gen * PSFC_GEN,
        "over_P":     P_gen > P_GEN_MAX,
    }
