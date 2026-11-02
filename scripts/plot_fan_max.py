"""
Maximum fan-only thrust (burner off) along the climb schedule, M 0–3.

    uv run python scripts/plot_fan_max.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnostics import max_fan_thrust
from mission import climb_altitude, thrust_required
from config import CLIMB_SCHEDULE, CDA_0, M_FLOOR

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
SERIES_2 = "#eb6834"


def sweep(n=301):
    rows = []
    for m in np.linspace(0.0, 3.0, n):
        h = float(climb_altitude(m))
        r = max_fan_thrust(float(m), h)
        r["F_req"] = float(thrust_required(m, h))
        rows.append(r)
    return rows


def crossover(rows):
    """First Mach above M_FLOOR where F_req exceeds the fan-only maximum."""
    M = np.array([r["M"] for r in rows])
    margin = np.array([r["F"] - r["F_req"] for r in rows])
    for i in range(1, len(M)):
        if M[i - 1] >= M_FLOOR and margin[i - 1] >= 0 > margin[i]:
            return M[i - 1] + margin[i - 1] / (margin[i - 1] - margin[i]) * (M[i] - M[i - 1])
    return None


def plot(rows, path):
    M = np.array([r["M"] for r in rows])
    F = np.array([r["F"] for r in rows]) / 1e3
    F_req = np.array([r["F_req"] for r in rows]) / 1e3
    pi_f = np.array([r["pi_f"] for r in rows])
    binding = sorted({r["binding"] for r in rows})
    M_x = crossover(rows)

    plt.rcParams.update({
        "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2,
    })
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    fig.patch.set_facecolor(SURFACE)

    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        for m, _ in CLIMB_SCHEDULE[1:-1]:
            ax.axvline(m, color=GRID, linewidth=0.8, linestyle="--", zorder=0)

    ax1.plot(M, F, color=SERIES, linewidth=2, label="Fan-only maximum")
    ax1.plot(M, F_req, color=SERIES_2, linewidth=2, label="Required, F_req")
    ax1.set_ylabel("Thrust [kN]")
    ax1.set_ylim(0, None)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9), frameon=False,
               labelcolor=INK_2, fontsize=9)
    if M_x is not None:
        F_x = np.interp(M_x, M, F)
        ax1.plot(M_x, F_x, "o", color=INK, markersize=7,
                 markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        ax1.annotate(f"fan alone insufficient\nabove M {M_x:.2f}", (M_x, F_x),
                     textcoords="offset points", xytext=(10, -40),
                     ha="left", color=INK, fontsize=9)
    for m, h in CLIMB_SCHEDULE[1:]:
        ax1.annotate(f"{h/1e3:.0f} km", (m, ax1.get_ylim()[1]), xytext=(3, -12),
                     textcoords="offset points", color=INK_2, fontsize=8)

    ax2.plot(M, pi_f, color=SERIES, linewidth=2)
    ax2.set_ylabel("π_f at limit")
    ax2.set_xlabel("Flight Mach number (on climb schedule)")
    ax2.set_xlim(0, 3)

    fig.suptitle("Fan-only maximum thrust vs requirement", x=0.08, ha="left",
                 fontsize=12, fontweight="bold")
    fig.text(0.08, 0.925,
             f"Burner off · 3.6 MW motor limit ({', '.join(binding)}-limited) · "
             f"A_fan 0.5 m² · CDA_0 {CDA_0:.3f} m² calibrated at M 0.6",
             ha="left", color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def table(rows):
    print(f"{'M':>4} {'h [m]':>7} {'pi_f':>6} {'F_max':>7} {'F_req':>7} {'ratio':>6} "
          f"{'T25 [K]':>7} {'TSFC [kg/N/h]':>13}  binding")
    for m in (0.0, 0.3, 0.6, 0.7, 0.8, 1.05, 1.5, 2.0, 3.0):
        r = min(rows, key=lambda r: abs(r["M"] - m))
        print(f"{r['M']:4.2f} {r['h']:7.0f} {r['pi_f']:6.3f} {r['F']/1e3:7.2f} "
              f"{r['F_req']/1e3:7.2f} {r['F_req']/r['F']:6.2f} {r['T25']:7.0f} "
              f"{r['TSFC']*3600:13.4f}  {r['binding']}")


if __name__ == "__main__":
    rows = sweep()
    out = os.path.join(ROOT, "figures", "fan_only_max_thrust.png")
    plot(rows, out)
    table(rows)
    print(f"\nF_req > F_max (fan alone insufficient) above M {crossover(rows):.3f}")
    print(f"saved {out}")
