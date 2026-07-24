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
def setup(seeds: list[int]):
    """Build the NQ (question, answer) parquet and the embed cache once."""
    import numpy as np
    import pandas as pd
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
    n_nt = 50 - 10
    used = set()
    for s in seeds:
        rng = np.random.default_rng(s)
        idx = rng.choice(len(all_answers), size=n_nt, replace=False)
        used.update(all_answers[i] for i in idx)
    strings = sorted(used) + [QUINE_TEXT, IDK_TEXT] + [str(i) for i in range(401)]
    strings = sorted(set(strings))
    print(f"embedding {len(strings)} strings")

    from sentence_transformers import SentenceTransformer
    from controlcharts.embedding import MODEL_ID, MODEL_REVISION
    os.environ["HF_HOME"] = "/cache"
    model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION,
                                trust_remote_code=True, cache_folder="/cache")
    embs = model.encode(strings, batch_size=128, convert_to_numpy=True,
                        normalize_embeddings=False).astype("float32")
    import pickle
    with open(CACHE, "wb") as f:
        pickle.dump({s: embs[i] for i, s in enumerate(strings)}, f, protocol=4)
    vol.commit()
    print(f"wrote cache {len(strings)} -> {CACHE}")


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
    }


@app.function(volumes={VOL: vol, "/cache": hf_cache}, timeout=1800, memory=8192)
def run_cell(args: tuple):
    N, k, seed = args
    r = _run_cell(N, k, seed)
    print(f"{r['N']} k={r['k']} s={r['seed']}: inf={r['infected_frac']:.2f} "
          f"intr={r['intrusion']} det={r['detected']}")
    return r


@app.local_entrypoint()
def test_one(n: int = 1000, k: str = "6", seed: int = 42):
    setup.remote([42])
    r = run_cell.remote((n, k, seed))
    print("RESULT:", json.dumps(r, indent=2))


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
