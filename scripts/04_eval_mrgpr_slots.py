import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from controller_D_slot_corrected import SlotCorrectedController
from control.controller_mrgpr_slots import MRGPBlockController
from models.residual_slots_gp import ResidualYSlotsGP, ResidualR2GP
from simulation import simulate


def _metrics(pos, x_refs, forces, conds, target, max_force):
    final_err = float(np.linalg.norm(pos[-1] - target))
    rms_err = float(np.sqrt(np.mean((pos - x_refs) ** 2)))
    sat = float(np.mean((forces >= max_force - 1e-6).any(axis=1)))
    cond_mean = float(np.mean(conds))
    cond_violate = float(np.mean(conds > 1e3))
    return final_err, rms_err, sat, cond_mean, cond_violate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gp-y", type=str, required=True)
    p.add_argument("--gp-r", type=str, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--hold", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--improve", type=float, default=0.1)
    p.add_argument("--yaw-rate", type=float, default=0.35, help="z-axis yaw rate [rad/s]")
    p.add_argument(
        "--target-min", type=float, default=0.9, help="minimum target position per axis"
    )
    p.add_argument(
        "--target-max", type=float, default=1.6, help="maximum target position per axis"
    )
    # MR-GPR trust tuning knobs (conservative defaults)
    p.add_argument("--trust-y", type=float, default=0.35, help="trust scale for Δy slots (0~1)")
    p.add_argument("--trust-r", type=float, default=0.25, help="trust scale for Δξ2 (0~1)")
    p.add_argument("--clip-y", type=float, default=0.3, help="variance clip for Δy (smaller=more conservative)")
    p.add_argument("--clip-r", type=float, default=0.1, help="variance clip for Δξ2")
    p.add_argument("--no-y", action="store_true", help="disable Δy correction")
    p.add_argument("--no-r", action="store_true", help="disable Δξ2 correction")
    p.add_argument(
        "--ignore-var-trust",
        action="store_true",
        help="ignore variance-based gating; use fixed trust scales",
    )
    p.add_argument("--cond-aware", action="store_true")
    p.add_argument("--cond-ref", type=float, default=1e3)
    p.add_argument("--ood-aware", action="store_true")
    p.add_argument("--ood-ref", type=float, default=3.0)
    p.add_argument("--cap-y-ratio", type=float, default=0.5)
    p.add_argument("--cap-y-abs", type=float, default=2.0)
    p.add_argument("--cap-r-ratio", type=float, default=0.5)
    p.add_argument("--cap-r-abs", type=float, default=0.5)
    # reporting
    p.add_argument("--report", action="store_true", help="pretty-print eval summary and failure reasons")
    p.add_argument("--out-json", type=str, default="artifacts/eval_metrics.json")
    p.add_argument("--episodes-csv", type=str, default="artifacts/eval_episode_metrics.csv")
    p.add_argument("--debug-json", type=str, default="", help="write per-episode MRGP stats")
    p.add_argument(
        "--offline-apply-json",
        type=str,
        default="",
        help="path to offline_apply_report.json for bookkeeping",
    )
    args = p.parse_args()

    gp_y = ResidualYSlotsGP.load(args.gp_y)
    gp_r = ResidualR2GP.load(args.gp_r)

    rng = np.random.default_rng(args.seed)

    metrics = {"baseline": [], "mrgp": []}
    cond_hist = {"baseline": [], "mrgp": []}
    episode_rows = []  # collect per-episode metrics for CSV
    debug_rows = []
    for _ in range(args.episodes):
        target = rng.uniform(args.target_min, args.target_max, size=3)
        omega_refs = [np.array([0.0, 0.0, args.yaw_rate])] * args.steps

        quad_base = SlotCorrectedController(gp_fx=None, gp_fr=None)
        pos_b, forces_b, _, _, x_refs_b, _, conds_b, _ = simulate(
            args.steps, target=target, hold_steps=args.hold, omega_refs=omega_refs, quad=quad_base
        )
        m_b = _metrics(pos_b, x_refs_b, forces_b, conds_b, target, quad_base.max_force)
        metrics["baseline"].append(m_b)
        cond_hist["baseline"].append(conds_b)
        episode_rows.append(("baseline",) + m_b)

        ts_y = 0.0 if args.no_y else args.trust_y
        ts_r = 0.0 if args.no_r else args.trust_r
        quad_mrgp = MRGPBlockController(
            gp_y=gp_y,
            gp_r=gp_r,
            trust_scale_y=ts_y,
            trust_scale_r=ts_r,
            var_clip_y=args.clip_y,
            var_clip_r=args.clip_r,
            ignore_var_trust=args.ignore_var_trust,
            cond_aware=args.cond_aware,
            cond_ref=args.cond_ref,
            ood_aware=args.ood_aware,
            ood_ref=args.ood_ref,
            cap_y_ratio=args.cap_y_ratio,
            cap_y_abs=args.cap_y_abs,
            cap_r_ratio=args.cap_r_ratio,
            cap_r_abs=args.cap_r_abs,
        )
        pos_m, forces_m, _, _, x_refs_m, _, conds_m, _ = simulate(
            args.steps, target=target, hold_steps=args.hold, omega_refs=omega_refs, quad=quad_mrgp
        )
        m_m = _metrics(pos_m, x_refs_m, forces_m, conds_m, target, quad_mrgp.max_force)
        metrics["mrgp"].append(m_m)
        cond_hist["mrgp"].append(conds_m)
        episode_rows.append(("mrgp",) + m_m)
        debug_rows.append(quad_mrgp.get_debug_stats())

    metrics_avg = {}
    for key, vals in metrics.items():
        arr = np.array(vals)
        metrics_avg[key] = {
            "final_err": float(np.mean(arr[:, 0])),
            "rms_err": float(np.mean(arr[:, 1])),
            "sat_pct": float(np.mean(arr[:, 2])),
            "cond_mean": float(np.mean(arr[:, 3])),
            "cond_violate": float(np.mean(arr[:, 4])),
        }

    cond_percentiles = {}
    for key, arrs in cond_hist.items():
        concat = np.concatenate(arrs) if arrs else np.array([])
        if concat.size:
            cond_percentiles[key] = np.percentile(concat, [50, 90, 99])
        else:
            cond_percentiles[key] = np.array([float("nan")]*3)
    if debug_rows:
        keys = debug_rows[0].keys()
        debug_summary = {k: float(np.mean([d[k] for d in debug_rows])) for k in keys}
    else:
        debug_summary = {}

    os.makedirs("artifacts", exist_ok=True)
    report = {
        "baseline": metrics_avg["baseline"],
        "mrgp": metrics_avg["mrgp"],
        "cond_percentiles": {k: cond_percentiles[k].tolist() for k in cond_percentiles},
        "config": {
            "episodes": args.episodes,
            "steps": args.steps,
            "hold": args.hold,
            "yaw_rate": args.yaw_rate,
            "target_min": args.target_min,
            "target_max": args.target_max,
            "trust_y": args.trust_y,
            "trust_r": args.trust_r,
            "clip_y": args.clip_y,
            "clip_r": args.clip_r,
            "ignore_var_trust": args.ignore_var_trust,
            "cond_aware": args.cond_aware,
            "cond_ref": args.cond_ref,
            "ood_aware": args.ood_aware,
            "ood_ref": args.ood_ref,
            "cap_y_ratio": args.cap_y_ratio,
            "cap_y_abs": args.cap_y_abs,
            "cap_r_ratio": args.cap_r_ratio,
            "cap_r_abs": args.cap_r_abs,
            "seed": args.seed,
            "offline_apply_json": args.offline_apply_json,
        },
        "debug_summary": debug_summary,
    }
    with open(args.out_json, "w") as fh:
        json.dump(report, fh)
    if args.debug_json:
        with open(args.debug_json, "w") as fh:
            json.dump({"episodes": debug_rows, "summary": debug_summary}, fh)

    # write per-episode CSV
    try:
        import csv
        with open(args.episodes_csv, "w", newline="") as cf:
            w = csv.writer(cf)
            w.writerow(["set", "final_err", "rms_err", "sat_pct", "cond_mean", "cond_violate"])
            for row in episode_rows:
                w.writerow(row)
    except Exception as e:
        print(f"[WARN] failed to write episodes CSV: {e}")

    b = metrics_avg["baseline"]
    m = metrics_avg["mrgp"]
    improve = (b["rms_err"] - m["rms_err"]) / max(b["rms_err"], 1e-9)
    sat_increase = m["sat_pct"] - b["sat_pct"]
    cond_increase = m["cond_violate"] - b["cond_violate"]

    if args.report:
        def _pct(x: float) -> str:
            return f"{100.0 * x:.1f}%"
        print("\n=== MR-GPR Evaluation (summary) ===")
        print(f"Baseline:  final_err={b['final_err']:.3f}  rms_err={b['rms_err']:.3f}  sat={_pct(b['sat_pct'])}  cond_mean={b['cond_mean']:.1f}  cond_viol={_pct(b['cond_violate'])}")
        print(f"MR-GPR:    final_err={m['final_err']:.3f}  rms_err={m['rms_err']:.3f}  sat={_pct(m['sat_pct'])}  cond_mean={m['cond_mean']:.1f}  cond_viol={_pct(m['cond_violate'])}")
        qb = cond_percentiles["baseline"]
        qm = cond_percentiles["mrgp"]
        print(
            f"Baseline condA p50/p90/p99: {qb[0]:.1f}/{qb[1]:.1f}/{qb[2]:.1f}"
        )
        print(
            f"MR-GPR  condA p50/p90/p99: {qm[0]:.1f}/{qm[1]:.1f}/{qm[2]:.1f}"
        )
        print(
            f"MR-GPR applied on {100.0 * debug_summary.get('applied_ratio', 0.0):.1f}% of steps"
        )
        print(f"→ rms improvement={100 * improve:.1f}%  Δsat={_pct(sat_increase)}  ΔcondViol={_pct(cond_increase)}")

    pass_cond = (
        improve >= args.improve
        and sat_increase <= 0.05
        and cond_increase <= 0.05
    )
    if pass_cond:
        print("EVAL PASS")
    else:
        print("EVAL FAIL")
        if args.report:
            reasons = []
            if improve < args.improve:
                reasons.append(f"rms improvement {100 * improve:.1f}% < target {100 * args.improve:.1f}%")
            if sat_increase > 0.05:
                reasons.append(f"sat increase {_pct(sat_increase)} > 5%")
            if cond_increase > 0.05:
                reasons.append(f"cond-violate increase {_pct(cond_increase)} > 5%")
            if reasons:
                print("Reasons:", "; ".join(reasons))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

