"""Run the (N, density) phase sweep on Modal — no tower dependency.

Each (N, mean_degree, seed) cell is an independent Modal container invocation,
so the shared-box OOM problem can't occur. All data is reconstructed off-tower:
the NQ question/answer text comes from HuggingFace (fastsim uses only the text,
never the parquet's embeddings), and the response-alphabet embedding cache is
built once into a Modal Volume.

Entrypoints:
  modal run modal_phase.py::setup            # build NQ parquet + embed cache in the Volume
  modal run modal_phase.py::test_one         # validate a single cell end-to-end
  modal run modal_phase.py::sweep            # fan out the full grid, write results locally

The sweep writes cells/*.json + an aggregated results file locally so the
existing phase_plot.py renders the diagram unchanged.
"""
import json
import os
import sys

import modal

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy", "pandas", "pyarrow", "scipy", "scikit-learn",
        "graspologic>=3.0.0", "hyppo>=0.4.0", "typing-extensions",
        "sentence-transformers==3.4.1", "torch==2.6.0", "einops==0.8.1",
        "pyyaml", "pydantic>=2", "datasets",
    )
    .env({"PYTHONPATH": "/app/src:/app/maps",
          "FASTSIM_EMBED_CACHE": "/vol/embed_cache.pkl"})
    .add_local_dir(os.path.join(REPO, "src"), "/app/src")
    .add_local_dir(os.path.join(REPO, "maps"), "/app/maps")
)

app = modal.App("controlcharts-phase", image=image)
vol = modal.Volume.from_name("controlcharts-phase-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

VOL = "/vol"
PARQUET = f"{VOL}/nq.parquet"
CACHE = f"{VOL}/embed_cache.pkl"


@app.function(volumes={VOL: vol, "/cache": hf_cache}, timeout=3600,
              memory=8192)
def setup(seeds: list[int], total_questions: int = 50, n_temporal: int = 10):
    """Build the NQ (question, answer) parquet and the embed cache once."""
    import numpy as np
    import pandas as pd
    import pickle
    from pathlib import Path

    if not Path(PARQUET).exists():
        from datasets import load_dataset
        ds = load_dataset("sentence-transformers/natural-questions", split="train",
                          cache_dir="/cache")
        pd.DataFrame({"question": ds["query"], "answer": ds["answer"]}).to_parquet(PARQUET)
        print(f"wrote {PARQUET} rows={len(ds['query'])}")
    else:
        print(f"{PARQUET} exists")

    # Build embed cache for the given seeds (replicate runner sampling).
    sys.path.insert(0, "/app/src")
    from controlcharts.fastsim.core import QUINE_TEXT, IDK_TEXT
    df = pd.read_parquet(PARQUET, columns=["answer"])
    all_answers = df["answer"].astype(str).tolist()
    n_nt = total_questions - n_temporal
    used = set()
    for s in seeds:
        rng = np.random.default_rng(s)
        idx = rng.choice(len(all_answers), size=n_nt, replace=False)
        used.update(all_answers[i] for i in idx)
    strings = sorted(used) + [QUINE_TEXT, IDK_TEXT] + [str(i) for i in range(401)]
    # Merge with the existing cache so runs at different Q / seed sets never
    # invalidate each other's alphabet.
    existing = {}
    if Path(CACHE).exists():
        with open(CACHE, "rb") as f:
            existing = pickle.load(f)
    strings = sorted(set(strings) - set(existing))
    if not strings:
        print(f"cache already covers all strings ({len(existing)} entries)")
        return
    print(f"embedding {len(strings)} new strings ({len(existing)} cached)")

    from sentence_transformers import SentenceTransformer
    from controlcharts.embedding import MODEL_ID, MODEL_REVISION
    os.environ["HF_HOME"] = "/cache"
    model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION,
                                trust_remote_code=True, cache_folder="/cache")
    embs = model.encode(strings, batch_size=128, convert_to_numpy=True,
                        normalize_embeddings=False).astype("float32")
    existing.update({s: embs[i] for i, s in enumerate(strings)})
    with open(CACHE, "wb") as f:
        pickle.dump(existing, f, protocol=4)
    vol.commit()
    print(f"wrote cache {len(existing)} total ({len(strings)} new) -> {CACHE}")


def _run_cell(N, k, seed):
    """Body shared by test_one and sweep. Returns the classification dict."""
    import numpy as np
    import yaml
    from pathlib import Path
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/app/maps")
    os.environ["HF_HOME"] = "/cache"
    from controlcharts.fastsim import fast_tdkps  # noqa: F401 (patches maps)
    from controlcharts.fastsim.phase_cell import make_config, iso_mirror, detected, INTRUSION_FRAC
    from controlcharts.fastsim.runner import run_fastsim

    mean_degree = None if (k == "full") else float(k)
    kstr = "full" if mean_degree is None else str(k)
    name = f"phase-N{N}-k{kstr}-s{seed}"
    work = Path(f"/tmp/{name}"); work.mkdir(parents=True, exist_ok=True)
    cfg = make_config(N, mean_degree, seed, name)
    cfg_path = work / "cfg.yaml"
    yaml.dump(cfg, open(cfg_path, "w"), sort_keys=False)

    run_dir = run_fastsim(str(cfg_path), data_path=PARQUET,
                          output_base=str(work / "runs"), panel_size=50)
    summ = json.load(open(run_dir / "results_summary.json"))
    infected_frac = summ["history"][-1]["infected_agents"] / N
    steps, iso = iso_mirror(run_dir)
    det = detected(steps, iso) if steps is not None else None
    intrusion = infected_frac > INTRUSION_FRAC
    return {
        "N": N, "k": kstr,
        "density": (mean_degree / (N - 1)) if mean_degree else 1.0,
        "mean_degree": mean_degree if mean_degree else (N - 1),
        "seed": seed, "infected_frac": infected_frac,
        "intrusion": bool(intrusion), "detected": det,
        "undetected_intrusion": bool(intrusion and det is False),
        # persist the iso-mirror trajectory so the detector can be re-scored
        # (different chart / k) without re-running the simulation
        "iso_steps": [int(s) for s in steps] if steps is not None else None,
        "iso_values": [float(v) for v in iso] if iso is not None else None,
    }


@app.function(volumes={VOL: vol, "/cache": hf_cache}, timeout=1800, memory=8192)
def run_cell(args: tuple):
    N, k, seed = args
    r = _run_cell(N, k, seed)
    print(f"{r['N']} k={r['k']} s={r['seed']}: inf={r['infected_frac']:.2f} "
          f"intr={r['intrusion']} det={r['detected']}")
    return r


@app.function(volumes={VOL: vol, "/cache": hf_cache}, timeout=5400, memory=32768)
def run_cell_big(args: tuple):
    """Same as run_cell but generous timeout/memory for very large N."""
    import time
    N, k, seed = args
    t0 = time.time()
    r = _run_cell(N, k, seed)
    r["wall_seconds"] = round(time.time() - t0, 1)
    print(f"{r['N']} k={r['k']} s={r['seed']}: inf={r['infected_frac']:.3f} "
          f"intr={r['intrusion']} det={r['detected']} ({r['wall_seconds']}s)")
    return r


@app.local_entrypoint()
def test_one(n: int = 1000, k: str = "6", seed: int = 42):
    setup.remote([42])
    r = run_cell.remote((n, k, seed))
    print("RESULT:", json.dumps(r, indent=2))


@app.function(volumes={VOL: vol, "/cache": hf_cache}, timeout=21600, memory=32768)
def run_search(spec: dict):
    """Run one cell with arbitrary N / degree / adversary params; return
    infected fraction + iso-mirror trajectory + accuracy for later scoring.
    spec keys: N (default 5), degree (mean degree; N-1 or None => full mesh),
    seed, adversary, prop, duration, ..."""
    import logging
    # Without this the root logger sits at WARNING and every logger.info in the
    # sim (ER graph build, 25/50/75/100% progress milestones with live
    # infection counts) is dropped -- Modal logs show only the setup prints, so
    # a long cell is completely opaque while it runs.
    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s %(name)s %(message)s")
    import sys
    from pathlib import Path
    sys.path.insert(0, "/app/src"); sys.path.insert(0, "/app/maps")
    os.environ["HF_HOME"] = "/cache"
    from controlcharts.fastsim import fast_tdkps  # noqa: F401
    from controlcharts.fastsim.phase_cell import (make_config, iso_mirror,
                                                  cell_name, INTRUSION_FRAC)
    from controlcharts.fastsim.runner import run_fastsim
    import yaml
    s = spec
    N = s.get("N", 5)
    deg = s.get("degree", None)
    tq = s.get("total_questions", 50)
    # degree >= N-1 (or None) is a complete graph -> use the full-mesh path
    # (an explicit CSR would be O(N^2) edges and blow up at large N)
    mean_degree = None if (deg is None or float(deg) >= N - 1) else float(deg)
    af = s.get("adv_frac") or None
    an = s.get("adv_n") or None
    name = cell_name(s)
    work = Path(f"/tmp/{name}"); work.mkdir(parents=True, exist_ok=True)
    cfg = make_config(N, mean_degree, s["seed"], name, prop_prob=s["prop"],
                      duration=s["duration"], max_p=s.get("max_p", 0.75),
                      shape=s.get("shape", 1.0), adversary=s["adversary"],
                      forget=s.get("forget", "decay"),
                      decay_coefficient=s.get("decay_coefficient", 0.05),
                      n_temporal=s.get("n_temporal", 10),
                      total_questions=tq,
                      questions_per_agent=s.get("questions_per_agent", 8),
                      adv_frac=af, adv_n=an)
    cp = work / "cfg.yaml"; yaml.dump(cfg, open(cp, "w"), sort_keys=False)
    run_dir = run_fastsim(str(cp), data_path=PARQUET, output_base=str(work / "runs"),
                          panel_size=50)
    summ = json.load(open(run_dir / "results_summary.json"))
    hist = summ["history"]
    inf = hist[-1]["infected_agents"] / N
    # victims-only infection: with a scaled adversary population the
    # adversaries themselves would floor infected_frac at adv_frac.
    last = hist[-1]
    inf_v = (last["infected_victims"] / last["n_victims"]
             if last.get("n_victims") else None)
    n_adv = int(N - last["n_victims"]) if last.get("n_victims") is not None else None
    # infection trajectory (was previously discarded -- needed to tell a
    # saturated attack from one still climbing when the sim ends)
    inf_traj = [(h["step"], h["infected_agents"]) for h in hist[::10]]
    steps, iso = iso_mirror(run_dir)
    # temporal / non-temporal accuracy vs time (mean over agents)
    from controlcharts.tdkps_analysis import _load_accuracy_from_snapshots
    _, acc_steps, _, t_acc, nt_acc = _load_accuracy_from_snapshots(run_dir / "snapshots")
    import numpy as _np
    t_series = t_acc.mean(axis=1) if t_acc is not None else None
    nt_series = nt_acc.mean(axis=1) if nt_acc is not None else None
    # end-of-sim accuracy = mean over the final 10% of snapshots
    end = (_np.asarray(acc_steps) >= 1800) if acc_steps is not None else None
    result = {**s, "N": N, "degree": deg, "infected_frac": inf,
              "infected_frac_victims": inf_v, "n_adversaries": n_adv,
              "infected_traj": inf_traj,
              "temporal_acc_end": float(t_series[end].mean()) if t_series is not None else None,
              "nontemporal_acc_end": float(nt_series[end].mean()) if nt_series is not None else None,
              "acc_steps": [int(x) for x in acc_steps] if acc_steps is not None else None,
              "temporal_acc": [float(x) for x in t_series] if t_series is not None else None,
              "nontemporal_acc": [float(x) for x in nt_series] if nt_series is not None else None,
              "iso_steps": [int(x) for x in steps] if steps is not None else None,
              "iso_values": [float(x) for x in iso] if iso is not None else None}
    # persist to the Volume so a dropped stream can't lose long runs (recover
    # via `sync`/collect); results/ keys include N/degree/variant/seed.
    rd = Path(f"{VOL}/results"); rd.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(rd / f"{name}.json", "w"))
    vol.commit()
    return result


@app.local_entrypoint()
def test_n(n: int = 500000, degree: int = 99, seed: int = 42, qscale: int = 1,
           adv_frac: float = 0.0):
    """Feasibility/timing probe for a single large-N attack cell."""
    import time
    setup.remote([seed], 50 * qscale, 10 * qscale)
    t0 = time.time()
    r = run_search.remote({"N": n, "degree": degree, "prop": 0.8, "duration": 1600,
                           "adversary": True, "seed": seed,
                           "total_questions": 50 * qscale, "n_temporal": 10 * qscale,
                           "questions_per_agent": 8 * qscale,
                           **({"adv_frac": adv_frac} if adv_frac else {})})
    v = r.get("infected_frac_victims")
    vstr = f"victims={v:.3f} " if v is not None else ""
    print(f"N={n} deg={degree} Q={50*qscale} adv={r.get('n_adversaries')}: "
          f"infected={r['infected_frac']:.3f} {vstr}"
          f"nt_acc_end={r['nontemporal_acc_end']:.3f} wall={time.time()-t0:.0f}s")


@app.local_entrypoint()
def detect_grid(reps: int = 3, ns: str = "5,10,100,1000,10000,100000",
                out: str = "/tmp/detect_grid.json", qscale: int = 1,
                adv_frac: float = 0.0, adv_n: int = 0, resume: bool = False):
    """Detectability vs empirical baseline over (N x mean-degree), d1600 slow
    schedule. Each cell runs BOTH noadv (baseline) and adv (attack) variants,
    reps each, and records iso-mirror + end-of-sim temporal/non-temporal acc.
    Degrees are only run where valid (degree < N).

    qscale multiplies the question-set size (Q=50*qscale) while holding the
    temporal:static composition (20%:80%), temporal_change_probability, and
    questions_per_turn constant; questions_per_agent scales with Q so the
    seeded fraction of the pool is unchanged.

    adv_frac>0 scales the adversary population with the network (e.g. 0.2 =>
    20% of agents are adversarial at every N) instead of the paper's single
    adversary. adv_n>0 instead fixes an ABSOLUTE cohort size (e.g. 10
    adversaries at every N, so their share shrinks as N grows)."""
    Ns = [int(x) for x in ns.split(",")]
    # decade steps; 9999 matters twice -- it is the full mesh at N=10000 (which
    # otherwise has no full-mesh cell, since 99999 >= N is skipped) and the
    # missing density step at N=100000.
    degrees = [4, 9, 99, 999, 9999, 99999]
    seed_list = list(range(42, 42 + reps))
    setup.remote(seed_list, 50 * qscale, 10 * qscale)
    qkw = {} if qscale == 1 else {"total_questions": 50 * qscale,
                                  "n_temporal": 10 * qscale,
                                  "questions_per_agent": 8 * qscale}
    if adv_frac:
        qkw["adv_frac"] = adv_frac
    if adv_n:
        qkw["adv_n"] = adv_n
    specs = []
    for N in Ns:
        for d in degrees:
            if d >= N:            # degree must be < N
                continue
            for s in seed_list:
                for adv in (False, True):
                    specs.append({"N": N, "degree": d, "prop": 0.8, "duration": 1600,
                                  "adversary": adv, "seed": s, **qkw})
    total = len(specs)
    if resume:
        # A detached map that loses its parent leaves the grid part-finished
        # (Modal keeps only the last triggered function alive). Cells are
        # written individually, so skip the ones already on the Volume.
        from controlcharts.fastsim.phase_cell import cell_name
        done = set(list_result_names.remote())
        specs = [s for s in specs if cell_name(s) not in done]
        print(f"resume: {total - len(specs)} of {total} cells already on the Volume")
    print(f"fanning out {len(specs)} cells "
          f"(of {total}; {total//(reps*2)} (N,degree) pairs x {reps} reps x 2 variants) "
          f"Q={50*qscale} adversaries={adv_n or (adv_frac and f'{adv_frac:g}xN') or 'single'}")
    if not specs:
        print("nothing to do"); return
    results = []
    for r in run_search.map(specs, order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception):
            print("cell failed:", r); continue
        results.append(r)
    json.dump(results, open(out, "w"))
    print(f"WROTE {len(results)} cells -> {out}")


@app.local_entrypoint()
def accuracy_trace(reps: int = 15, out: str = "/tmp/accuracy_trace.json"):
    """Temporal & non-temporal accuracy vs time at N=5 full mesh, for the
    baseline (noadv) and d1600 (slow attack) for contrast."""
    seed_list = list(range(42, 42 + reps))
    setup.remote(seed_list)
    specs = []
    for s in seed_list:
        specs.append({"prop": 0.8, "duration": 1600, "adversary": False, "seed": s})   # baseline
        specs.append({"prop": 0.8, "duration": 1600, "adversary": True, "seed": s})    # d1600
    results = []
    for r in run_search.map(specs, order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception):
            print("cell failed:", r); continue
        results.append(r)
    json.dump(results, open(out, "w"))
    print(f"WROTE {len(results)} cells -> {out}")


@app.local_entrypoint()
def noise_sources(reps: int = 15, out: str = "/tmp/noise_sources.json"):
    """Isolate what drives the noadv alarm floor at N=5 full mesh: forgetting
    (decay) and/or the temporal questions. Runs noadv under 4 combinations."""
    seed_list = list(range(42, 42 + reps))
    setup.remote(seed_list)
    variants = [
        ("decay+temporal", "decay", 10),   # current baseline
        ("none+temporal", "none", 10),     # no forgetting
        ("decay+notemporal", "decay", 0),  # no temporal churn
        ("none+notemporal", "none", 0),    # fully static
    ]
    specs = []
    for label, forget, n_temp in variants:
        for s in seed_list:
            specs.append({"prop": 0.8, "duration": 1600, "adversary": False,
                          "seed": s, "forget": forget, "n_temporal": n_temp,
                          "label": label})
    print(f"fanning out {len(specs)} noadv variant cells")
    results = []
    for r in run_search.map(specs, order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception):
            print("cell failed:", r); continue
        results.append(r)
    json.dump(results, open(out, "w"))
    print(f"WROTE {len(results)} cells -> {out}")


@app.local_entrypoint()
def fig4_conditions(reps: int = 20, out: str = "/tmp/fig4_conditions.json"):
    """Paper's four Figure-4 conditions at N=5 full mesh (prop=0.8): noadv,
    d100 (fast), d800, d1600 (slow). Returns iso-mirror trajectories for the
    empirical-alarm-rate-vs-alpha analysis."""
    seed_list = list(range(42, 42 + reps))
    setup.remote(seed_list)
    specs = []
    for s in seed_list:
        specs.append({"prop": 0.8, "duration": 1600, "adversary": False, "seed": s})  # noadv
        for d in (100, 800, 1600):
            specs.append({"prop": 0.8, "duration": d, "adversary": True, "seed": s})
    print(f"fanning out {len(specs)} cells (4 conditions x {reps} reps)")
    results = []
    for r in run_search.map(specs, order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception):
            print("cell failed:", r); continue
        results.append(r)
    json.dump(results, open(out, "w"))
    print(f"WROTE {len(results)} cells -> {out}")


@app.local_entrypoint()
def config_search(seeds: int = 10, out: str = "/tmp/config_search.json"):
    """Search near the Figure-5 family for an N=5 full-mesh config where the
    slow attack infects everyone yet evades the adaptive chart. Sweeps
    propagation probability (takeover speed) x duration, plus a noadv baseline."""
    seed_list = list(range(42, 42 + seeds))
    setup.remote(seed_list)
    props = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    durations = [1600, 3200]
    specs = [{"prop": p, "duration": d, "adversary": True, "seed": s}
             for p in props for d in durations for s in seed_list]
    specs += [{"prop": 0.8, "duration": 1600, "adversary": False, "seed": s}
              for s in seed_list]   # noadv baseline
    print(f"fanning out {len(specs)} search cells on Modal")
    results = []
    for r in run_search.map(specs, order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception):
            print("cell failed:", r); continue
        results.append(r)
    json.dump(results, open(out, "w"))
    print(f"WROTE {len(results)} search cells -> {out}")


@app.local_entrypoint()
def validate_full5(reps: int = 10, out: str = "/tmp/fig4_full5.json"):
    """Paper Figure 4 d1600 validation: N=5, fully connected, slow schedule,
    no-LLM path. Returns iso-mirror trajectories for all reps so the adaptive
    control chart can be scored at any k locally."""
    seed_list = list(range(42, 42 + reps))
    setup.remote(seed_list)
    combos = [(5, "full", s) for s in seed_list]
    results = []
    for r in run_cell.map(combos, order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception):
            print("cell failed:", r); continue
        results.append(r)
    json.dump(results, open(out, "w"))
    print(f"WROTE {len(results)} reps -> {out}")


@app.local_entrypoint()
def test_scale(n: int = 100000, k: str = "12", seed: int = 42):
    """Feasibility test for a very large network."""
    setup.remote([seed])
    r = run_cell_big.remote((n, k, seed))
    print("RESULT:", json.dumps(r, indent=2))


@app.function(volumes={VOL: vol, "/cache": hf_cache}, timeout=3600, memory=16384)
def run_cell_long(args: tuple):
    """Longer timeout / more memory for large-N cells (e.g. N=100k ~24min).

    Persists the result to the Volume so a dropped local streaming connection
    can't lose completed work — recover with the `collect` function.
    """
    import time
    from pathlib import Path
    N, k, seed = args
    t0 = time.time()
    r = _run_cell(N, k, seed)
    r["wall_seconds"] = round(time.time() - t0, 1)
    rd = Path(f"{VOL}/results"); rd.mkdir(parents=True, exist_ok=True)
    nm = f"phase-N{r['N']}-k{r['k']}-s{r['seed']}"
    json.dump(r, open(rd / f"{nm}.json", "w"))
    vol.commit()
    print(f"{r['N']} k={r['k']} s={r['seed']}: inf={r['infected_frac']:.3f} "
          f"intr={r['intrusion']} det={r['detected']} ({r['wall_seconds']}s)")
    return r


@app.function(volumes={VOL: vol}, timeout=600)
def list_result_names():
    """Names (stem, no .json) of every cell already persisted to the Volume.
    Lets `detect_grid --resume` skip finished work after an interrupted map."""
    from pathlib import Path
    rd = Path(f"{VOL}/results")
    if not rd.exists():
        return []
    return sorted(p.stem for p in rd.glob("*.json"))


@app.function(volumes={VOL: vol}, timeout=600)
def collect():
    """Return every result persisted to the Volume (drop-proof recovery)."""
    import glob
    vol.reload()
    out = []
    for f in glob.glob(f"{VOL}/results/*.json"):
        try:
            out.append(json.load(open(f)))
        except Exception:
            pass
    return out


@app.local_entrypoint()
def warm_cache(reps: int = 30, qscale: int = 1):
    """Build the embed cache for seeds 42..42+reps at one Q, on its own.

    `setup` is a read-modify-write of a single pickle, so grids launched
    concurrently with seeds not yet cached can drop each other's strings; the
    losers then fall back to embedding in-cell, loading a SentenceTransformer
    per container. Warm each Q the grids will use first and their own setup
    calls become no-ops.
    """
    setup.remote(list(range(42, 42 + reps)), 50 * qscale, 10 * qscale)


@app.local_entrypoint()
def sync():
    """Pull all Volume-persisted results into the local cells dir."""
    from pathlib import Path
    out = Path(os.path.join(REPO, "experiments", "phase_modal", "cells"))
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in collect.remote():
        nm = f"phase-N{r['N']}-k{r['k']}-s{r['seed']}"
        json.dump(r, open(out / f"{nm}.json", "w"))
        n += 1
    print(f"synced {n} results from Volume -> {out}")


@app.local_entrypoint()
def fill_triangle(seeds: int = 30, out_dir: str = None):
    """Fill the (N, density) cells between mean_degree 14 and full mesh.

    Adds intermediate degrees at every N where the degree is valid (< N),
    so the upper-left triangle of the diagram is populated instead of a
    sparse full-mesh staircase. Writes into the shared phase_modal cells dir.
    """
    from pathlib import Path
    Ns = [100, 300, 1000, 3000, 10000, 100000]
    degs = [20, 50, 100, 300, 1000, 3000, 10000, 30000]
    seed_list = list(range(42, 42 + seeds))
    out = Path(out_dir or os.path.join(REPO, "experiments", "phase_modal"))
    (out / "cells").mkdir(parents=True, exist_ok=True)
    all_combos = [(N, str(d), s) for N in Ns for d in degs if d < N for s in seed_list]
    # resumable: skip cells whose result json already exists
    combos = [c for c in all_combos
              if not (out / "cells" / f"phase-N{c[0]}-k{c[1]}-s{c[2]}.json").exists()]
    setup.remote(seed_list)
    print(f"fanning out {len(combos)}/{len(all_combos)} triangle-fill cells "
          f"({len(all_combos) - len(combos)} already done) on Modal")
    done = 0
    for r in run_cell_long.map(combos, order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception):
            print("cell failed:", r); continue
        nm = f"phase-N{r['N']}-k{r['k']}-s{r['seed']}"
        json.dump(r, open(out / "cells" / f"{nm}.json", "w"))
        done += 1
    print(f"done: {done}/{len(combos)} triangle-fill cells -> {out}")


@app.local_entrypoint()
def sweep_n(n: int = 100000, seeds: int = 30, out_dir: str = None):
    """Full density x seed sweep for a SINGLE N. Writes into the shared
    phase_modal cells dir so it merges with the existing N<=10000 grid."""
    from pathlib import Path
    ks = ["0.5", "1", "1.5", "2", "3", "4", "6", "8", "12", "14", "full"]
    # drop degrees that can't exist on n nodes (mean degree must be < n)
    ks = [k for k in ks if k == "full" or float(k) < n]
    seed_list = list(range(42, 42 + seeds))
    out = Path(out_dir or os.path.join(REPO, "experiments", "phase_modal"))
    (out / "cells").mkdir(parents=True, exist_ok=True)
    all_combos = [(n, k, s) for k in ks for s in seed_list]
    combos = [c for c in all_combos
              if not (out / "cells" / f"phase-N{c[0]}-k{c[1]}-s{c[2]}.json").exists()]
    setup.remote(seed_list)
    print(f"fanning out {len(combos)}/{len(all_combos)} cells at N={n} "
          f"({len(all_combos) - len(combos)} already done) on Modal")

    done = 0
    for r in run_cell_long.map(combos, order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception):
            print("cell failed:", r); continue
        nm = f"phase-N{r['N']}-k{r['k']}-s{r['seed']}"
        json.dump(r, open(out / "cells" / f"{nm}.json", "w"))
        done += 1
    print(f"done: {done}/{len(combos)} cells at N={n} -> {out}")


@app.local_entrypoint()
def sweep(out_dir: str = None, seeds: int = 30):
    import itertools
    from pathlib import Path
    Ns = [100, 300, 1000, 3000, 10000]
    ks = ["0.5", "1", "1.5", "2", "3", "4", "6", "8", "12", "14", "full"]
    seed_list = list(range(42, 42 + seeds))
    setup.remote(seed_list)
    combos = [(N, k, s) for N, k, s in itertools.product(Ns, ks, seed_list)]
    print(f"fanning out {len(combos)} cells on Modal")

    out = Path(out_dir or os.path.join(REPO, "experiments", "phase_modal"))
    (out / "cells").mkdir(parents=True, exist_ok=True)
    results = []
    for r in run_cell.map(combos, order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception):
            print("cell failed:", r); continue
        results.append(r)
        nm = f"phase-N{r['N']}-k{r['k']}-s{r['seed']}"
        json.dump(r, open(out / "cells" / f"{nm}.json", "w"))
    json.dump(sorted(results, key=lambda c: (c["N"], c["mean_degree"], c["seed"])),
              open(out / "cells_all.json", "w"), indent=1)
    print(f"done: {len(results)}/{len(combos)} cells -> {out}")
