# Quadrotor MR-GPR Simulation

This repository contains a small Python example of a discrete-time quadrotor model controlled with Model Reference Gaussian Process Regression (MR-GPR) ideas. The code estimates thrust and torques, maps them to rotor forces, and advances the vehicle state. The main script also plots the $x$, $y$, and $z$ positions over time.

## Requirements
- Python 3.12+
- `numpy`
- `matplotlib`

Install dependencies:

```bash
pip install numpy matplotlib
```

## Running the simulation
Execute the main script to simulate the quadrotor for 200 steps, print a few sample states, and generate `trajectory.png` plotting the position components versus time:

```bash
python simulation.py
```

Orientation references `R_ref` are produced by tilting the body $z$-axis toward
the total desired acceleration `a_ref - g` at each step.  The associated angular
rates are derived from the difference between successive orientation references.
The `simulate` helper returns the position and force histories, attitude errors,
the actual rotation matrices, and the corresponding `R_ref` matrices.  Custom
angular-rate profiles may still be supplied to override this default behaviour.

### Custom orientation example

The snippet below commands a constant yaw rate and prints the final reference
orientation returned by `simulate`:

```python
import numpy as np
from simulation import simulate

steps = 100
omega_refs = [np.array([0.0, 0.0, 0.5])] * steps
_, _, _, _, R_refs = simulate(steps, omega_refs=omega_refs)
print(R_refs[-1])
```

A smooth cubic trajectory is still used so that the quadrotor starts and stops
at rest while moving from the origin to the goal.

## Testing
The project uses `pytest` for testing (no tests are currently implemented). Run:

```bash
pytest
```

