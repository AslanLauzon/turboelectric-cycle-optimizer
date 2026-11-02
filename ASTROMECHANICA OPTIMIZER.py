import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # ASTROMECHANICA ENGINE OPTIMIZER
    An over eager (potential) intern's optimizer for your sick ass engine.

    Minimum-fuel cycle analysis of the Duality turboelectric engine. At each Mach
    number on the climb schedule, the optimizer picks the fan pressure ratio π_f,
    the burner exit temperature T4 and the generator power that meet the thrust
    requirement with the least total fuel. It considers three modes: **fan only**,
    **fan + burner**, and **ram** (fan off). Only CDA_0 is calibrated (at M 0.6),
    so the mode transitions are predictions.
    """)
    return


@app.cell
def _(mo):
    import sys

    _root = str(mo.notebook_dir())
    if _root not in sys.path:
        sys.path.insert(0, _root)

    import csv
    import numpy as np
    import matplotlib.pyplot as plt
    from optimizer import optimize_point
    from mission import climb_altitude
    import config

    root = _root
    return climb_altitude, config, csv, np, optimize_point, plt, root


@app.cell
def _(config, mo):
    mo.md(f"""
    ### Key assumptions (all in `config.py`)

    | Parameter | Value | Status |
    |---|---|---|
    | Generator max power | {config.P_GEN_MAX / 1e6:.1f} MW | public (CT7-based demonstrator) |
    | Fan motor limit | {config.P_FAN_MAX_CONT / 1e6:.1f} MW (4 × 900 kW) | Gen 5 design target |
    | Generator PSFC | {config.PSFC_GEN * 3.6e6:.2f} kg/kWh | assumed |
    | Fan-face area | {config.A_FAN} m² | assumed |
    | Burner max T4 | {config.T_BURNER_MAX:.0f} K | assumed |
    | Drag area CDA_0 | {config.CDA_0:.3f} m² | calibrated at M 0.6 |
    """)
    return


@app.cell
def _(mo):
    mach = mo.ui.slider(0.0, 3.0, step=0.05, value=1.5, label="Flight Mach",
                        debounce=True, show_value=True, full_width=True)
    mach
    return (mach,)


@app.cell
def _(climb_altitude, mach, mo, optimize_point):
    _M = float(mach.value)
    _h = float(climb_altitude(_M))
    with mo.status.spinner(title="Optimizing…"):
        point = optimize_point(_M, _h)

    _names = {"fan_only": "Fan only", "fan_burner": "Fan + burner", "ram": "Ram"}
    _T4 = "off" if point["T4"] is None else f"{point['T4']:.0f} K"
    _rows = "\n".join(
        f"| {_names[k]} | {'infeasible' if v == float('inf') else f'{v:.4f}'} |"
        for k, v in point["branch_fuel"].items()
    )
    mo.md(f"""
    ### M {_M:.2f} at {_h / 1e3:.1f} km → **{_names[point['mode']]}**

    | Required thrust | π_f | T4 | P_gen | Total fuel | Fuel / F_req |
    |---|---|---|---|---|---|
    | {point['F_req'] / 1e3:.1f} kN | {point['pi_f']:.3f} | {_T4} | {point['P_gen'] / 1e6:.2f} MW | {point['total_fuel']:.3f} kg/s | {point['total_fuel'] / point['F_req'] * 3600:.4f} kg/(N·h) |

    Cheapest fuel flow each mode can manage at this point [kg/s]:

    | Mode | Fuel |
    |---|---|
    {_rows}
    """)
    return


@app.cell
def _(csv, mo, np, root):
    import os

    _path = os.path.join(root, "results", "mach_sweep.csv")
    mo.stop(not os.path.exists(_path),
            mo.md("Run `uv run python scripts/plot_sweep.py` to generate the sweep."))

    def _f(x):
        return np.nan if x in ("", "None") else float(x)

    with open(_path) as _fh:
        _rows = list(csv.DictReader(_fh))
    sweep = {k: np.array([_f(r[k]) for r in _rows])
             for k in ("M", "F_req_N", "P_gen_W", "pi_f", "fuel_kg_s",
                       "fuel_turbojet", "fuel_fixed_pi")}
    sweep["mode"] = [r["mode"] for r in _rows]
    transitions = [0.5 * (sweep["M"][i] + sweep["M"][i + 1])
                   for i in range(len(_rows) - 1)
                   if sweep["mode"][i] != sweep["mode"][i + 1]]
    return sweep, transitions


@app.cell
def _(mach, plt, sweep, transitions):
    _ink, _ink2, _grid = "#0b0b0b", "#52514e", "#e4e3df"
    _M = sweep["M"]
    _k = 3600.0 / sweep["F_req_N"]

    _fig, (_a1, _a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 3]})
    for _ax in (_a1, _a2):
        _ax.grid(axis="y", color=_grid, linewidth=0.8)
        for _s in ("top", "right"):
            _ax.spines[_s].set_visible(False)
        for _m in transitions:
            _ax.axvline(_m, color=_ink2, linewidth=0.8, linestyle="--")
        _ax.axvline(mach.value, color=_ink, linewidth=1.5)

    _a1.plot(_M, sweep["P_gen_W"] / 1e6, color="#2a78d6", linewidth=2)
    _a1.set_ylabel("P_gen [MW]")
    _a1.set_ylim(0, None)

    _a2.plot(_M, sweep["fuel_fixed_pi"] * _k, color="#eb6834", linewidth=2,
             label="Fixed-π_f turbofan")
    _a2.plot(_M, sweep["fuel_turbojet"] * _k, color="#1baf7a", linewidth=2,
             label="Always-burning")
    _a2.plot(_M, sweep["fuel_kg_s"] * _k, color="#2a78d6", linewidth=2,
             label="Optimized, mode-switching")
    _a2.set_ylabel("Fuel / F_req [kg/(N·h)]")
    _a2.set_xlabel("Flight Mach number (on climb schedule)")
    _a2.set_ylim(0, None)
    _a2.set_xlim(0, 3)
    _a2.legend(loc="lower right", frameon=False, fontsize=9)
    _a1.set_title("Precomputed sweep — dashed: mode transitions, solid: selected Mach",
                  loc="left", fontsize=10, color=_ink2)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
