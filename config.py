"""
All assumed/fixed parameters for the Duality optimizer.
Each entry notes its source or justification.

SOURCES
  [1] Mattingly, Elements of Gas Turbine Propulsion, 2nd ed.
  [2] Aviation Week, Jan 2026 — Duality ground demonstrator (GE CT7 base, 4 MW gen)
  [3] Helix SPX242-94 datasheet, Dec 2025
  [4] Astro Mechanica founder interview, 2024 (ramjet Tt ceiling)
  [5] MIL-E-5007D — inlet recovery standard
  [6] NACA 1135 / [1] — Jet-A thermochemistry
  [ASSUMED] — no public data; flagged for sensitivity analysis
  [TARGET]  — announced design target, not demonstrated hardware
  [CALIBRATED] — set once from a stated anchor condition
"""

# ── Atmosphere ────────────────────────────────────────────────────────────────
GAMMA   = 1.4          # specific heat ratio, air (cold-gas approx below ~1000 K) [1]
R_AIR   = 287.058      # J/(kg·K), specific gas constant for air
CP_AIR  = 1005.0       # J/(kg·K), cp at low temperature
G       = 9.80665      # m/s²

# ── Fuel (Jet-A) ──────────────────────────────────────────────────────────────
LHV     = 43.2e6       # J/kg, lower heating value of Jet-A [6]

# ── Turbogenerator ────────────────────────────────────────────────────────────
P_GEN_MAX    = 4.0e6   # W, peak electrical output [2]
PSFC_GEN     = 0.30 / 3.6e6  # kg/J (= 0.30 kg/(kW·h) ≈ 8.33e-8 kg/J), turboshaft
                        # power-specific fuel consumption [ASSUMED]. Typical small
                        # turboshaft 0.25–0.35 kg/(kW·h); flag for sensitivity.

# ── Electrical chain ──────────────────────────────────────────────────────────
ETA_GENERATOR = 0.98   # generator efficiency [2 implied]
ETA_MOTOR     = 0.98   # motor efficiency [2 implied]
ETA_CHAIN     = ETA_GENERATOR * ETA_MOTOR  # ≈ 0.9604

# ── Propulsor motors (Gen 5) ──────────────────────────────────────────────────
# One engine = one turbogen + one propulsor. Gen 4 hardware (Helix SPX242-94,
# 4 × 300 kW [3]) gives only 1.2 MW, which can't absorb the 4 MW generator.
# Gen 5 values are DESIGN TARGETS, not tested hardware.
# η_chain·P_GEN_MAX ≈ 3.84 MW ≥ 3.6 MW, so the chain is roughly balanced.
N_MOTORS           = 4          # [TARGET]
P_MOTOR_CONTINUOUS = 900e3      # W per motor, Gen 5 design target [TARGET]
P_FAN_MAX_CONT     = N_MOTORS * P_MOTOR_CONTINUOUS  # 3.6 MW total continuous

# ── Fan ───────────────────────────────────────────────────────────────────────
ETA_FAN_POLY = 0.88    # [ASSUMED] polytropic efficiency, 2-stage electric fan
                        # Typical axial fan: 0.85–0.92; 0.88 is mid-range.
PI_F_MIN     = 1.05    # [ASSUMED] minimum useful fan pressure ratio
PI_F_MAX     = 3.5     # [ASSUMED] maximum fan PR before stall/choking becomes
                        # the dominant constraint at high Mach
A_FAN        = 0.5     # m², [ASSUMED] fan-face annulus flow area (~0.87 m tip
                        # dia at hub/tip 0.4). ṁ = MFP(M_FACE)·A_FAN·P02/√T02 [1]
                        # → ~96 kg/s SL static, ~106 kg/s at M3/20 km.
                        # Sensitivity sweep in step 7.
M_FACE       = 0.55    # [ASSUMED] fan-face axial Mach, typical 0.5–0.6. Held
                        # constant (no fan map) until step 7.

# ── Burner ────────────────────────────────────────────────────────────────────
ETA_BURNER   = 0.97    # combustion efficiency [ASSUMED] typical for aircraft burners
DELTA_P_BURNER = 0.04  # fractional total-pressure loss across burner [ASSUMED] 3–5% typical
T_BURNER_MAX = 2100.0  # K, max burner exit temperature [ASSUMED]
                        # Stoichiometric ~2540 K; practical limit set by materials/NOx.
                        # Ramjet inlet Tt ceiling 1350 K [4] is a separate constraint.
T_FAN_EXIT_MAX = 900.0 # K, fan-exit temperature limit [ASSUMED]
                        # Protects motor insulation; generous for now.

# ── Mission ───────────────────────────────────────────────────────────────────
CLIMB_SCHEDULE = (     # (M, h [m]) breakpoints, linear between [ASSUMED]
    (0.0, 0.0),
    (0.3, 0.0),
    (0.8, 3_000.0),
    (1.5, 11_000.0),
    (3.0, 20_000.0),
)

# Thrust requirement F_req = q·CDA_0·k(M), floored at low speed.
DRAG_RISE = (          # (M, k) breakpoints, transonic drag rise, linear between [ASSUMED]
    (0.0, 1.0),
    (0.8, 1.0),
    (1.05, 2.5),
    (1.5, 2.0),
    (2.0, 1.6),
    (3.0, 1.3),
)
F_REQ_FLOOR = 8.0e3    # N, minimum F_req below M_FLOOR (takeoff/climb excess) [ASSUMED]
M_FLOOR     = 0.3
CDA_0       = 0.4470   # m², [CALIBRATED] once: F_req = 0.75·F_max(fan-only) at
                        # M 0.6 on the climb schedule (diagnostics.calibrate_cda0).
                        # Only calibrated value in the model; light-off and fan
                        # power-down Mach are predictions. Recalibrate if the
                        # propulsor model or A_FAN changes (test_mission guards it).

# ── Nozzle ────────────────────────────────────────────────────────────────────
# Ideally expanded at first (step 1–5); variable nozzle area added in step 7.
ETA_NOZZLE = 0.98      # [ASSUMED] nozzle velocity coefficient squared ≈ 0.96 typical
