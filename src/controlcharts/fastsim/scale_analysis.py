"""Headline figures for the scale grid.

Figure A (scale_collapse.png): mean known/agent and infected-agent fraction
vs step, one curve per N — well-behaved intensive scaling shows collapse of
the knowledge curves and the epidemic takeoff shifting with N.

Figure B (scale_detection.png): iso-mirror trajectories with the adaptive
control band per N, plus detection latency vs N. Control-band parameters
match figure3.py (calibration end 300, window 200 iters, k=2).

Usage:
  python -m controlcharts.fastsim.scale_analysis --runs 'experiments/results/scale-nollm-*_fast_*'
"""

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CALIBRATION_END = 300
WINDOW_SIZE_ITERS = 200
K = 2
ATTACK_START = 400


def parse_name(run_dir: Path):
    m = re.match(r"scale-nollm-N(\d+)-(adv|clean)-s(\d+)_fast_", run_dir.name)
    if not m:
        return None
    return int(m.group(1)), m.group(2), int(m.group(3))


def load_history(run_dir: Path):
    with open(run_dir / "results_summary.json") as f:
        d = json.load(f)
    h = d["history"]
    return (np.array([r["step"] for r in h]),
            np.array([r["mean_known"] for r in h]),
            np.array([r["infected_agents"] for r in h]) / d["config"]["agents"]["count"])


def load_isomirror(run_dir: Path):
    from sklearn.manifold import Isomap
    npz_files = glob.glob(str(run_dir / "*_tdkps_embeddings.npz"))
    if not npz_files:
        return None, None
    d = np.load(npz_files[0], allow_pickle=True)
    em, steps = d["embedding_matrix"], d["steps"]
    flat = em.reshape(em.shape[0], -1)
    dist = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=2)
    iso = Isomap(n_components=1, metric="precomputed",
                 n_neighbors=min(5, len(steps) - 1)).fit_transform(dist)[:, 0]
    iso = iso - iso[0]
    if iso[-1] < 0:
        iso = -iso
    return np.asarray(steps), iso


def control_bars(steps, iso):
    """Port of figure3.compute_control_bars: sliding window mean +- K std."""
    interval = steps[1] - steps[0] if len(steps) > 1 else 10
    window = max(1, WINDOW_SIZE_ITERS // interval)
    cal_end = int(np.searchsorted(steps, CALIBRATION_END))
    upper = np.full(len(steps), np.nan)
    lower = np.full(len(steps), np.nan)
    for i in range(cal_end, len(steps)):
        w = iso[max(cal_end, i - window):i]
        if len(w) > 1:
            m, s = np.mean(w), np.std(w)
        elif len(w) == 1:
            m, s = w[0], 0.0
        else:
            continue
        upper[i], lower[i] = m + K * s, m - K * s
    return upper, lower, cal_end


def detection_step(steps, iso):
    """First step at/after attack start where iso exits the adaptive band."""
    upper, lower, _ = control_bars(steps, iso)
    for i in range(len(steps)):
        if steps[i] < ATTACK_START or np.isnan(upper[i]):
            continue
        if iso[i] > upper[i] or iso[i] < lower[i]:
            return int(steps[i])
    return None


def false_alarms(steps, iso):
    """Out-of-band points on a clean run (post-calibration)."""
    upper, lower, cal_end = control_bars(steps, iso)
    n = 0
    for i in range(cal_end, len(steps)):
        if not np.isnan(upper[i]) and (iso[i] > upper[i] or iso[i] < lower[i]):
            n += 1
    return n, len(steps) - cal_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="experiments/results/scale-nollm-*_fast_*")
    ap.add_argument("--out-prefix", default="experiments/figures/scale")
    args = ap.parse_args()

    groups = defaultdict(list)   # (N, arm) -> [run_dir]
    for d in sorted(glob.glob(args.runs)):
        p = Path(d)
        key = parse_name(p)
        if key:
            groups[(key[0], key[1])].append(p)
    sizes = sorted({k[0] for k in groups})
    print(f"found runs: { {k: len(v) for k, v in sorted(groups.items())} }")
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)

    colors = {n: c for n, c in zip(sizes, ["C0", "C1", "C2", "C3"])}

    # ---- Figure A: collapse + epidemic ------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for n_agents in sizes:
        for arm, ls in (("clean", "--"), ("adv", "-")):
            hs = [load_history(d) for d in groups.get((n_agents, arm), [])]
            if not hs:
                continue
            steps = hs[0][0]
            known = np.stack([h[1] for h in hs])
            inf = np.stack([h[2] for h in hs])
            axes[0].plot(steps, known.mean(0), ls, color=colors[n_agents],
                         label=f"N={n_agents} {arm}", alpha=0.9)
            if arm == "adv":
                axes[1].plot(steps, inf.mean(0), color=colors[n_agents], label=f"N={n_agents}")
                axes[1].fill_between(steps, inf.min(0), inf.max(0), color=colors[n_agents], alpha=0.15)
    axes[0].set_title("Mean known questions / agent (collapse across N)")
    axes[0].set_xlabel("step"); axes[0].legend(fontsize=8)
    axes[1].set_title("Infected-agent fraction (adv arm)")
    axes[1].axvline(ATTACK_START, color="gray", ls=":", label="attack start")
    axes[1].set_xlabel("step"); axes[1].legend(fontsize=8)

    # quine fraction in snapshot responses
    for n_agents in sizes:
        rows = []
        for d in groups.get((n_agents, "adv"), []):
            metas = sorted(glob.glob(str(d / "snapshots" / "snapshot_step_*_meta.json")))
            s, f = [], []
            for mp in metas:
                with open(mp) as fh:
                    m = json.load(fh)
                s.append(m["step"])
                flat = [r for row in m["responses"] for r in row]
                f.append(np.mean(["i lost the game" in r.lower() for r in flat]))
            rows.append((np.array(s), np.array(f)))
        if rows:
            steps = rows[0][0]
            ys = np.stack([y for _, y in rows])
            axes[2].plot(steps, ys.mean(0), color=colors[n_agents], label=f"N={n_agents}")
            axes[2].fill_between(steps, ys.min(0), ys.max(0), color=colors[n_agents], alpha=0.15)
    axes[2].set_title("Quine fraction in panel responses (adv arm)")
    axes[2].axvline(ATTACK_START, color="gray", ls=":")
    axes[2].set_xlabel("step"); axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_collapse.png", dpi=130)
    print(f"wrote {args.out_prefix}_collapse.png")

    # ---- Figure B: iso-mirror + detection ---------------------------------
    fig, axes = plt.subplots(2, len(sizes), figsize=(6 * len(sizes), 8), squeeze=False)
    latencies = {}
    fa_rates = {}
    for col, n_agents in enumerate(sizes):
        ax = axes[0][col]
        lats = []
        for d in groups.get((n_agents, "adv"), []):
            steps, iso = load_isomirror(d)
            if steps is None:
                continue
            upper, lower, _ = control_bars(steps, iso)
            ax.plot(steps, iso, color=colors[n_agents], alpha=0.7)
            ax.fill_between(steps, lower, upper, color="gray", alpha=0.15)
            det = detection_step(steps, iso)
            lats.append(det)
            if det is not None:
                ax.axvline(det, color="red", ls=":", alpha=0.5)
        latencies[n_agents] = lats
        ax.axvline(ATTACK_START, color="gray", ls=":")
        ax.set_title(f"N={n_agents} adv: iso-mirror + {K}$\\sigma$ band")

        ax = axes[1][col]
        fas = []
        for d in groups.get((n_agents, "clean"), []):
            steps, iso = load_isomirror(d)
            if steps is None:
                continue
            ax.plot(steps, iso, color=colors[n_agents], alpha=0.7)
            n_fa, n_pts = false_alarms(steps, iso)
            fas.append((n_fa, n_pts))
        fa_rates[n_agents] = fas
        ax.set_title(f"N={n_agents} clean: iso-mirror")
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_detection.png", dpi=130)
    print(f"wrote {args.out_prefix}_detection.png")

    summary = {}
    for n_agents in sizes:
        lats = latencies.get(n_agents, [])
        detected = [l for l in lats if l is not None]
        summary[n_agents] = {
            "runs": len(lats),
            "detected": len(detected),
            "latency_steps": [l - ATTACK_START for l in detected],
            "mean_latency": float(np.mean([l - ATTACK_START for l in detected])) if detected else None,
            "clean_false_alarm_points": [list(x) for x in fa_rates.get(n_agents, [])],
        }
        print(f"N={n_agents}: detected {len(detected)}/{len(lats)}, "
              f"latencies={summary[n_agents]['latency_steps']}, "
              f"clean FA={fa_rates.get(n_agents)}")
    with open(f"{args.out_prefix}_detection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
