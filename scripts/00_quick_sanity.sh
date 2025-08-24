#!/bin/bash
set -e
mkdir -p artifacts
python -m data.build_slots_td --runs 3 --steps 120 --hold 30 --seed 0 --out artifacts/slots_td.npz
python scripts/02_train_slots_gp.py --data artifacts/slots_td.npz --out-y artifacts/gp_y.pkl --out-r artifacts/gp_r.pkl
python scripts/03_verify_residuals.py --data artifacts/slots_td.npz --gp-y artifacts/gp_y.pkl --gp-r artifacts/gp_r.pkl
python scripts/04_eval_mrgpr_slots.py --gp-y artifacts/gp_y.pkl --gp-r artifacts/gp_r.pkl --episodes 5
