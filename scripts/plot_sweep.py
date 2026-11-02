"""
Step 6: minimum-fuel Mach sweep along the climb schedule, with baselines.

    uv run python scripts/plot_sweep.py

Writes figures/mach_sweep.png and results/mach_sweep.csv.
"""

import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sweep import (mach_sweep, find_transitions, power_down_onset,
                   fixed_pi_design, fixed_pi_point, _on_schedule)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BAND = "#f0efec"
C_OPT = "#2a78d6"
C_FIX = "#eb6834"
C_TJ = "#1baf7a"

MODE_LABEL = {"fan_only": "Fan only", "fan_burner": "Fan + burner", "ram": "Ram"}


def run(n=121):
    M = np.linspace(0.0, 3.0, n)
    t = [time.perf_counter()]
    opt = mach_sweep(M);                                          t.append(time.perf_counter())
    tj = mach_sweep(M, modes=("fan_burner",));                    t.append(time.perf_counter())
    pi_fix = fixed_pi_design(M)
    fix = [fixed_pi_point(*_on_schedule(m), pi_fix) for m in M];  t.append(time.perf_counter())
    trans = find_transitions(opt);                                t.append(time.perf_counter())
    dt = np.diff(t)
    print(f"time [s]: optimized {dt[0]:.1f}, turbojet {dt[1]:.1f}, "
          f"fixed-pi {dt[2]:.1f}, transitions {dt[3]:.1f}")
    return M, opt, tj, fix, pi_fix, trans


def col(rows, key, scale=1.0):
    return np.array([np.nan if r[key] is None else r[key] * scale for r in rows], dtype=float)


def tsfc_req(rows):
    """Fuel per required thrust [kg/(N·h)]; over-delivered thrust is not credited."""
    return np.array([r["total_fuel"] / r["F_req"] * 3600.0 for r in rows])


def plot(M, opt, tj, fix, pi_fix, trans, path):
    plt.rcParams.update({
        "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2,
    })
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 2, 3]})
    fig.patch.set_facecolor(SURFACE)

    edges = [0.0] + [t[0] for t in trans] + [3.0]
    modes = [trans[0][1]] + [t[2] for t in trans] if trans else [opt[0]["mode"]]

    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        for i in range(1, len(edges) - 1, 2):
            ax.axvspan(edges[i], edges[i + 1], color=BAND, zorder=0, linewidth=0)
        for m, _, _ in trans:
            ax.axvline(m, color=INK_2, linewidth=0.8, linestyle="--", zorder=1)

    ax_pi, ax_t4, ax_p, ax_tsfc = axes

    ax_pi.plot(M, col(opt, "pi_f"), color=C_OPT, linewidth=2)
    ax_pi.set_ylabel("π_f")
    ax_pi.set_ylim(0.95, None)
    top = ax_pi.get_ylim()[1]
    for a, b, mode in zip(edges, edges[1:], modes):
        ax_pi.text(0.5 * (a + b), top, MODE_LABEL[mode], ha="center", va="bottom",
                   color=INK, fontsize=9, fontweight="bold")
    for m, _, _ in trans:
        ax_pi.annotate(f"M {m:.2f}", (m, 1.0), xytext=(4, 2),
                       textcoords="offset points", color=INK_2, fontsize=8)

    ax_t4.plot(M, col(opt, "T4"), color=C_OPT, linewidth=2)
    ax_t4.set_ylabel("T4 [K]\n(burner lit)")

    ax_p.plot(M, col(opt, "P_gen", 1e-6), color=C_OPT, linewidth=2)
    ax_p.set_ylabel("P_gen [MW]")
    ax_p.set_ylim(0, None)
    M_on = power_down_onset(opt)
    if M_on is not None:
        P_on = np.interp(M_on, M, col(opt, "P_gen", 1e-6))
        ax_p.plot(M_on, P_on, "o", color=INK, markersize=7,
                  markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        ax_p.annotate(f"power-down onset\nM {M_on:.2f}", (M_on, P_on),
                      xytext=(-10, -30), textcoords="offset points",
                      ha="right", color=INK, fontsize=9)

    ax_tsfc.plot(M, tsfc_req(fix), color=C_FIX, linewidth=2,
                 label=f"Fixed π_f = {pi_fix:.2f} turbofan")
    ax_tsfc.plot(M, tsfc_req(tj), color=C_TJ, linewidth=2,
                 label="Always-burning (fan + burner only)")
    ax_tsfc.plot(M, tsfc_req(opt), color=C_OPT, linewidth=2,
                 label="Optimized, mode-switching")
    ax_tsfc.set_ylabel("Fuel / F_req [kg/(N·h)]")
    ax_tsfc.set_ylim(0, None)
    ax_tsfc.legend(loc="lower right", frameon=False, labelcolor=INK_2, fontsize=9)
    ax_tsfc.set_xlabel("Flight Mach number (on climb schedule)")
    ax_tsfc.set_xlim(0, 3)

    fig.suptitle("Minimum-fuel operating schedule", x=0.08, ha="left",
                 fontsize=12, fontweight="bold")
    fig.text(0.08, 0.948, "Total fuel (turbogen + burner) minimised at F = F_req · "
             "CDA_0 calibrated at M 0.6 only; transitions are predictions",
             ha="left", color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def write_csv(M, opt, tj, fix, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "h_m", "F_req_N", "mode", "pi_f", "T4_K", "P_gen_W",
                    "fuel_kg_s", "TSFC_kg_Ns", "fuel_fan_only", "fuel_fan_burner",
                    "fuel_ram", "fuel_turbojet", "fuel_fixed_pi"])
        for m, r, a, b in zip(M, opt, tj, fix):
            bf = r["branch_fuel"]
            w.writerow([f"{m:.4f}", f"{r['h']:.1f}", f"{r['F_req']:.2f}", r["mode"],
                        r["pi_f"], r["T4"], r["P_gen"], r["total_fuel"], r["TSFC"],
                        bf["fan_only"], bf["fan_burner"], bf["ram"],
                        a["total_fuel"], b["total_fuel"]])


def table(M, opt, tj, fix, trans):
    for m, a, b in trans:
        print(f"transition {MODE_LABEL[a]} → {MODE_LABEL[b]} at M {m:.3f}")
    print(f"power-down onset (P_gen < 99% limit) at M {power_down_onset(opt):.3f}"
          f" (sweep resolution {M[1] - M[0]:.3f})")
    t_opt, t_tj, t_fix = tsfc_req(opt), tsfc_req(tj), tsfc_req(fix)
    print(f"\n{'M':>4} {'mode':>12} {'pi_f':>6} {'T4':>6} {'P_gen':>6} "
          f"{'opt':>7} {'turbojet':>9} {'fixed pi':>9}   fuel/F_req [kg/(N·h)]")
    for target in (0.0, 0.3, 0.6, 0.8, 1.05, 1.5, 1.8, 2.0, 2.5, 3.0):
        i = int(np.argmin(abs(M - target)))
        r = opt[i]
        print(f"{M[i]:4.2f} {MODE_LABEL[r['mode']]:>12} {r['pi_f']:6.3f} "
              f"{'-' if r['T4'] is None else format(r['T4'], '6.0f'):>6} "
              f"{r['P_gen']/1e6:6.2f} {t_opt[i]:7.4f} {t_tj[i]:9.4f} {t_fix[i]:9.4f}")


if __name__ == "__main__":
    M, opt, tj, fix, pi_fix, trans = run()
    plot(M, opt, tj, fix, pi_fix, trans, os.path.join(ROOT, "figures", "mach_sweep.png"))
    write_csv(M, opt, tj, fix, os.path.join(ROOT, "results", "mach_sweep.csv"))
    table(M, opt, tj, fix, trans)
