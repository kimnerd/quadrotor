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

The simulation now computes a reference attitude `R_ref` from the instantaneous
acceleration at each step so the vehicle tilts toward the target position. A
smooth cubic trajectory is used so that the quadrotor starts and stops at rest
while moving from the origin to the goal.

## Testing
The project uses `pytest` for testing (no tests are currently implemented). Run:

```bash
pytest
```

