---
title: Duality Engine Optimizer
emoji: ✈️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Duality engine optimizer

A cycle analysis of Astro Mechanica's **Duality** turboelectric engine, built
from public data and documented assumptions (Mattingly, *Elements of Gas
Turbine Propulsion*).

**Question:** if you burn as little fuel as possible at every point of a
Mach 0–3 climb, do you get the published operating schedule?

**[Try the interactive demo →](https://huggingface.co/spaces/AslanL/turboelectric-cycle-optimizer)**

## How it works

- A **turbogenerator** powers an **electric fan**. A **burner** behind the fan
  can add thrust.
- At each Mach number the optimizer picks the fan pressure ratio, the burner
  temperature and the generator power that meet the required thrust with
  the least total fuel.
- It chooses between three modes: **fan only**, **fan + burner**, and
  **ram** (fan off, burner only).
- Every assumption and its source is in [`config.py`](config.py). Only one
  value, the aircraft drag area, is calibrated (at M 0.6). Everything else is
  a prediction.

## Results

![Mach sweep](figures/mach_sweep.png)

| Event | Model | Published |
|---|---|---|
| Burner lights | M 0.69 | ~0.8 |
| Fan starts powering down | M 1.80 | 1.7–2.0 |
| Fan fully off (ram) | M 2.05 | — |

- **Fan power-down is the real test.** The optimizer winds the fan down because
  electric-fan thrust gets more fuel-expensive with speed. Nothing was tuned
  to make that happen.
- **Burner light-off is weaker evidence.** It happens because the fan runs out of
  thrust, so it mostly follows from the calibration.
- Varying the main assumptions one at a time keeps power-down between
  **M 1.65 and 2.0**. Generator fuel efficiency matters most. See
  `scripts/sensitivity.py`.

## Limitations

There is no fan map and the nozzle is ideal. The duct is lossless when the fan
is off, which flatters ram mode. Generator efficiency is constant. The 3.6 MW
motor rating is a design target, not tested hardware.

## Run

```
uv run pytest tests/ -q                            # tests
uv run python scripts/plot_sweep.py                # results figure + CSV
uv run marimo run "ASTROMECHANICA OPTIMIZER.py"    # interactive demo
```
