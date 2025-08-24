# Quadrotor MR-GPR Simulation

This repository contains a small Python example of a discrete-time quadrotor model controlled with Model Reference Gaussian Process Regression (MR-GPR) ideas. The code estimates thrust and torques, maps them to rotor forces, and advances the vehicle state. The main script also plots the $x$, $y$, and $z$ positions over time.

## Requirements
- Python 3.12+
- `numpy`
- `matplotlib`
- `pillow` (for GIF animation)

Install dependencies:

```bash
pip install numpy matplotlib pillow
```

## Running the simulation
Execute the main script to simulate the quadrotor for 200 steps and then hold at the goal.  The script now trains a small MR-GPR inverse model and runs both the ideal inverse allocation and MR-GPR-based allocation, printing a few sample states and generating `trajectory.png` and an animated `trajectory.gif` comparing the reference, ideal, and MR-GPR 3-D paths:

```bash
python simulation.py
```

Orientation references `R_ref` are produced by tilting the body $z$-axis toward
the total desired acceleration `a_ref - g` at each step.  The associated angular
rates are derived from the difference between successive orientation references.
The `simulate` helper returns the position and force histories, attitude errors,
the actual rotation matrices, the translational references `x_refs`, and the
corresponding `R_ref` matrices.  Custom angular-rate profiles may still be
supplied to override this default behaviour.

PID gains for both position and attitude control are exposed as arguments to
the `Quadrotor` constructor. The defaults now use lighter proportional and
derivative gains on the horizontal $x$ and $y$ axes\—$k_p=0.1$ and
$k_d=0.3$\—which keep overshoot within roughly $10\%$ while leaving the $z$
response unchanged. For additional adjustments you can call the `tune_pid`
helper to perform a grid search over a range of PID coefficients, evaluating
the integrated tracking error, terminal error after a hold at the goal, and any
overshoot across the entire trajectory.

### Custom orientation example

The snippet below commands a constant yaw rate and prints the final reference
orientation returned by `simulate`:

```python
import numpy as np
from simulation import simulate

steps = 100
omega_refs = [np.array([0.0, 0.0, 0.5])] * steps
_, _, _, _, _, R_refs = simulate(steps, omega_refs=omega_refs)
print(R_refs[-1])
```

A smooth cubic trajectory is still used so that the quadrotor starts and stops
at rest while moving from the origin to the goal.

### Inspecting the default reference path

```python
from simulation import simulate

steps = 100
_, _, _, _, x_refs, R_refs = simulate(steps)
print(x_refs[-1])  # near [1, 1, 1]
print(R_refs[-1])  # final orientation reference
```

## Testing
The project uses `pytest` for testing (no tests are currently implemented). Run:

```bash
pytest
```

## Residual learning experiments

Two experimental scripts explore Gaussian–process residual corrections:

- `run_fxfr_residual_experiment.py` collects data and evaluates controllers
  using only the nominal plant.  Because the plant matches the model exactly,
  the learned residuals are typically near zero and results mirror the
  baseline.
- `run_experiment_slots_residual.py` separates the nominal model from a
  mismatched real plant and applies GP‑based slot corrections during block
  inverse control.  Use this script for experiments where residual learning
  should affect performance.

  Both scripts accept `--residual-data` and `--abs-data` arguments to load or
  save `.npz` datasets so training data can be reused across runs.

### Building slot-residual datasets

For the MR-GPR slot controller, a moderate dataset improves stability.  A
recommended starting point is:

```bash
python -m data.build_slots_td --runs 6 --steps 160 --hold 40 --seed 0 --out artifacts/slots_td.npz
```

The resulting `artifacts/slots_td.npz` can then be passed to
`scripts/02_train_slots_gp.py` and the downstream verification/evaluation
scripts.

If training is compute-constrained, a fast mode disables hyperparameter
restarts and uses an isotropic kernel. Subsampling further reduces cost:

```bash
python scripts/02_train_slots_gp.py --data artifacts/slots_td.npz \
  --out-y artifacts/gp_y.pkl --out-r artifacts/gp_r.pkl \
  --val-split 0.2 --seed 0 --fast --subsample 800
```

### Semi-fast training & conservative MR-GPR evaluation

```bash
python scripts/02_train_slots_gp.py --data artifacts/slots_td.npz \
  --out-y artifacts/gp_y.pkl --out-r artifacts/gp_r.pkl \
  --val-split 0.2 --seed 0 --restarts 3 --optimizer fmin_l_bfgs_b --calibrate --strict

python scripts/04_eval_mrgpr_slots.py \
  --gp-y artifacts/gp_y.pkl --gp-r artifacts/gp_r.pkl \
  --episodes 8 --steps 220 --hold 350 \
  --trust-y 0.6 --trust-r 0.35 --clip-y 1.0 --clip-r 0.3 \
  --report --out-json artifacts/eval_metrics.json --episodes-csv artifacts/eval_episode_metrics.csv
```

### Checking results (reports)

After training:
```bash
cat artifacts/gp_report.json | python - <<'PY'
import sys,json; r=json.load(sys.stdin)
print("VAL Δy RMSE=%.3f  Δξ2 RMSE=%.3f  Calib_y=%.3f  Calib_r=%.3f" % (r["rmse_y"], r["rmse_r"], r["calib_y"], r["calib_r"]))
PY
```

Residual verification (pretty summary + artifacts):
```bash
python scripts/03_verify_residuals.py \
  --data artifacts/slots_td.npz \
  --gp-y artifacts/gp_y.pkl --gp-r artifacts/gp_r.pkl \
  --report \
  --out-json artifacts/residual_report.json \
  --out-csv artifacts/residual_dims.csv
```
Artifacts:
- `artifacts/residual_report.json` — global metrics (RMSE/R2/calibration/correlations)
- `artifacts/residual_dims.csv` — per-dimension RMSE & R²

Evaluation (pretty summary + per-episode CSV):
```bash
python scripts/04_eval_mrgpr_slots.py \
  --gp-y artifacts/gp_y.pkl --gp-r artifacts/gp_r.pkl \
  --episodes 8 --steps 220 --hold 350 \
  --trust-y 0.6 --trust-r 0.35 --clip-y 1.0 --clip-r 0.3 \
  --report --out-json artifacts/eval_metrics.json --episodes-csv artifacts/eval_episode_metrics.csv
```
Artifacts:
- `artifacts/eval_metrics.json` — baseline vs MR-GPR averages
- `artifacts/eval_episode_metrics.csv` — per-episode metrics table

If evaluation fails, the script prints **reasons** (e.g., insufficient RMS improvement, increased saturation or conditioning violations) to guide parameter tuning.

# Ablation (which correction helps?)
python scripts/04_eval_mrgpr_slots.py ... --no-r   # Δy only
python scripts/04_eval_mrgpr_slots.py ... --no-y   # Δξ2 only

# Quick sweep for trust/clip (bash)
for TY in 0.4 0.6 0.8; do
  for CY in 0.3 0.6 1.0; do
    python scripts/04_eval_mrgpr_slots.py \
      --gp-y artifacts/gp_y.pkl --gp-r artifacts/gp_r.pkl \
      --episodes 8 --steps 220 --hold 350 \
      --trust-y $TY --trust-r 0.35 --clip-y $CY --clip-r 0.3 --report || true
  done
done

