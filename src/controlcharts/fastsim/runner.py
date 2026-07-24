"""Config-driven runner for FastSim.

Mirrors cli.py's setup RNG call structure (question selection, seeding
permutation) so a fastsim run with the same seed starts from the same
initial conditions as the original simulator. Snapshot output format is
byte-compatible with TemporalKernelHook so run_tdkps_analysis and the
figure scripts consume it unchanged.
"""

import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import Config
from ..temporal_questions import TEMPORAL_QUESTIONS
from .core import FastSim

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent.parent.parent / "data" / "nq_embedded.parquet"
DEFAULT_PANEL_SIZE = 100


def build_er_csr(N: int, mean_degree: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Undirected Erdos-Renyi G(N, p=mean_degree/(N-1)) as symmetric CSR
    neighbor lists. Samples ~N*mean_degree/2 candidate edges (sparse regime),
    drops self-loops and duplicates. Returns (indptr, indices)."""
    m = int(round(N * mean_degree / 2))
    if m <= 0:
        return np.zeros(N + 1, dtype=np.int64), np.zeros(0, dtype=np.int64)
    # oversample to offset self-loop/duplicate loss
    over = int(m * 1.2) + 8
    i = rng.integers(0, N, size=over)
    j = rng.integers(0, N, size=over)
    good = i != j
    i, j = i[good], j[good]
    # canonical undirected key to dedupe
    lo = np.minimum(i, j); hi = np.maximum(i, j)
    key = lo.astype(np.int64) * N + hi
    _, uniq = np.unique(key, return_index=True)
    i, j = i[uniq], j[uniq]
    if len(i) > m:
        sel = rng.choice(len(i), size=m, replace=False)
        i, j = i[sel], j[sel]
    # symmetric edge list
    src = np.concatenate([i, j])
    dst = np.concatenate([j, i])
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    indptr = np.zeros(N + 1, dtype=np.int64)
    np.add.at(indptr, src + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, dst.astype(np.int64)


def _load_embed_cache() -> dict[str, np.ndarray] | None:
    """Load a precomputed string->embedding cache if FASTSIM_EMBED_CACHE is set.

    Lets many parallel cells reuse one precomputed alphabet cache instead of
    each loading the ~2-3 GB nomic model (which OOMs the box at parallelism).
    Build it once with precompute_embed_cache.py.
    """
    import os
    import pickle
    path = os.environ.get("FASTSIM_EMBED_CACHE")
    if path and Path(path).exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _embed_alphabet(strings: list[str]) -> dict[str, np.ndarray]:
    """Embed the response alphabet. Uses FASTSIM_EMBED_CACHE if it covers every
    string (no model load); otherwise loads the model locally for the misses.
    """
    cache = _load_embed_cache()
    if cache is not None:
        missing = [s for s in strings if s not in cache]
        if not missing:
            return {s: cache[s] for s in strings}
        logger.warning(f"Embed cache missing {len(missing)}/{len(strings)} strings; "
                       f"loading model for the remainder")
    else:
        missing = strings

    from sentence_transformers import SentenceTransformer
    from ..embedding import MODEL_ID, MODEL_REVISION

    logger.info(f"Embedding {len(missing)} strings via model")
    model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True)
    embs = model.encode(missing, batch_size=64, show_progress_bar=False,
                        normalize_embeddings=False, convert_to_numpy=True).astype(np.float32)
    out = {s: cache[s] for s in strings if cache and s in cache} if cache else {}
    out.update({s: embs[i] for i, s in enumerate(missing)})
    return out


def run_fastsim(
    config_path: str,
    data_path: str | None = None,
    output_base: str = "experiments/results",
    panel_size: int = DEFAULT_PANEL_SIZE,
    skip_embed: bool = False,
    skip_analysis: bool = False,
) -> Path:
    config = Config.from_yaml(config_path)
    assert config.agents.use_llm is False, "fastsim only supports use_llm: false"

    name = config.experiment.name
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(output_base) / f"{name}_fast_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / f"config_{Path(config_path).name}")

    # --- data ---------------------------------------------------------------
    import pandas as pd
    dp = Path(data_path) if data_path else DEFAULT_DATA_PATH
    df = pd.read_parquet(dp, columns=["question", "answer"])
    all_questions = df["question"].tolist()
    all_answers = df["answer"].tolist()
    logger.info(f"Loaded {len(all_questions)} QA pairs from {dp}")

    N = config.agents.count
    n_temporal = min(config.data.n_temporal, len(TEMPORAL_QUESTIONS))
    n_nontemporal = min(config.data.total_questions - n_temporal, len(all_questions))

    # Same rng call sequence as cli.py: choice for questions in play, then
    # permutation for the seeding assignment.
    setup_rng = np.random.default_rng(config.simulation.seed)
    indices = setup_rng.choice(len(all_questions), size=n_nontemporal, replace=False)
    questions_nt = [all_questions[i] for i in indices]
    answers_nt = [all_answers[i] for i in indices]
    temporal_qs = TEMPORAL_QUESTIONS[:n_temporal]

    qpa = config.data.questions_per_agent
    shuffled = setup_rng.permutation(n_nontemporal)
    seeded = [shuffled[i * qpa: min(i * qpa + qpa, n_nontemporal)] for i in range(N)]

    defection = {}
    for c in config.agents.custom:
        if c.defection_schedule is not None:
            s = c.defection_schedule
            defection[c.id] = {"start": s.start, "duration": s.duration,
                               "max_p": s.max_p, "shape": s.shape}

    # Contact graph: full mesh (default) or Erdos-Renyi by mean degree.
    nbr_indptr = nbr_indices = None
    if config.network.topology == "er":
        k = config.network.mean_degree
        assert k is not None, "er topology requires network.mean_degree"
        graph_rng = np.random.default_rng(config.simulation.seed ^ 0x6E7)
        nbr_indptr, nbr_indices = build_er_csr(N, float(k), graph_rng)
        logger.info(f"ER graph: N={N} mean_degree~{k} "
                    f"(density={k/(N-1):.2e}), edges={len(nbr_indices)//2}")
    elif config.network.topology != "full_mesh":
        raise ValueError(f"fastsim supports full_mesh|er, got {config.network.topology}")

    sim = FastSim(
        n_agents=N,
        n_nontemporal=n_nontemporal,
        n_temporal=n_temporal,
        answers_nontemporal=answers_nt,
        seeded=seeded,
        retrieval_k=config.agents.retrieval_k,
        propagation_probability=config.agents.propagation_probability,
        cross_question_propagation=config.agents.cross_question_propagation,
        decay_coefficient=config.simulation.forget_strategy.decay_coefficient,
        forget_strategy=config.simulation.forget_strategy.strategy,
        temporal_change_probability=config.data.temporal_change_probability,
        defection_schedules=defection,
        questions_per_turn=config.simulation.questions_per_turn,
        seed=config.simulation.seed,
        nbr_indptr=nbr_indptr,
        nbr_indices=nbr_indices,
    )

    # --- probes and panel ---------------------------------------------------
    # Same equal-sampling rule as TemporalKernelHook: all temporal questions
    # plus an equal number of sampled non-temporal ones.
    hook_rng = np.random.default_rng(config.simulation.seed)
    if n_temporal > 0:
        n_probe_nt = min(n_temporal, n_nontemporal)
    else:
        tk = config.simulation.temporal_kernel
        n_probe_nt = min(tk.n_nontemporal_sample or tk.sample_size, n_nontemporal)
    probe_nt_idx = hook_rng.choice(n_nontemporal, size=n_probe_nt, replace=False)

    probe_qs = np.concatenate([
        np.arange(n_nontemporal, n_nontemporal + n_temporal, dtype=np.int64),  # temporal first
        np.asarray(sorted(probe_nt_idx), dtype=np.int64),
    ])
    probe_questions = temporal_qs + [questions_nt[i] for i in sorted(probe_nt_idx)]
    temporal_mask = [True] * n_temporal + [False] * n_probe_nt

    if N <= panel_size:
        panel = np.arange(N, dtype=np.int64)
    else:
        others = hook_rng.choice(np.arange(1, N), size=panel_size - 1, replace=False)
        panel = np.concatenate([[0], np.sort(others)]).astype(np.int64)
    logger.info(f"Panel: {len(panel)} agents, probes: {len(probe_qs)} questions "
                f"({n_temporal} temporal + {n_probe_nt} non-temporal)")

    snap_rng = np.random.default_rng(config.simulation.seed ^ 0xC0FFEE)
    snapshots_dir = run_dir / "snapshots"
    snapshots_dir.mkdir(exist_ok=True)
    pending = []

    def snapshot_cb(t: int):
        responses = sim.snapshot_responses(panel, probe_qs, t, snap_rng)
        pending.append({
            "step": t,
            "responses": responses,
            "temporal_values": {temporal_qs[i]: int(sim.temporal_values[i]) for i in range(n_temporal)},
        })

    # --- run ----------------------------------------------------------------
    logging.info(f"Running fastsim: N={N} Q={sim.Q} iters={config.simulation.max_iterations}")
    t0 = time.time()
    interval = config.simulation.temporal_kernel.interval if config.simulation.temporal_kernel.enabled else 0
    sim.run(config.simulation.max_iterations, snapshot_cb=snapshot_cb, snapshot_interval=interval)
    elapsed = time.time() - t0
    logger.info(f"Simulation done in {elapsed:.1f}s ({len(pending)} snapshots, "
                f"{sim.dropped_slots} dropped query slots)")

    # --- finalize snapshots -------------------------------------------------
    index = []
    alphabet = sorted({r for s in pending for row in s["responses"] for r in row})
    emb_table = None
    if not skip_embed:
        try:
            emb_table = _embed_alphabet(alphabet)
        except ImportError as e:
            logger.warning(f"sentence-transformers unavailable ({e}); writing metadata only")

    for s in pending:
        step = s["step"]
        meta_path = snapshots_dir / f"snapshot_step_{step:04d}_meta.json"
        npz_path = snapshots_dir / f"snapshot_step_{step:04d}.npz"
        A, P = len(panel), len(probe_qs)
        shape = [A, P, len(next(iter(emb_table.values()))) if emb_table else 0]
        with open(meta_path, "w") as f:
            json.dump({
                "step": step,
                "questions": probe_questions,
                "responses": s["responses"],
                "agent_ids": [int(x) for x in panel],
                "shape": shape,
                "temporal_mask": temporal_mask,
                "temporal_values": s["temporal_values"],
            }, f, indent=2)
        if emb_table is not None:
            embs = np.stack([
                np.stack([emb_table[r] for r in row]) for row in s["responses"]
            ]).astype(np.float32)
            np.savez_compressed(npz_path, embeddings=embs, step=step)
            index.append({"step": step, "path": str(npz_path),
                          "metadata_path": str(meta_path), "shape": list(embs.shape)})

    if index:
        with open(snapshots_dir / "snapshots_index.json", "w") as f:
            json.dump(index, f, indent=2)

    # --- summary ------------------------------------------------------------
    with open(run_dir / "results_summary.json", "w") as f:
        json.dump({
            "config": config.model_dump(),
            "timestamp": datetime.now().isoformat(),
            "elapsed_time_seconds": elapsed,
            "engine": "fastsim",
            "panel_agent_ids": [int(x) for x in panel],
            "dropped_query_slots": int(sim.dropped_slots),
            "history": sim.history,
            "final_known_count": [int(x) for x in sim.known_count],
            "final_infected_pairs": [int(x) for x in sim.inf_count],
        }, f)
    logger.info(f"Summary written to {run_dir / 'results_summary.json'}")

    if index and not skip_analysis:
        try:
            from ..tdkps_analysis import run_tdkps_analysis
            run_tdkps_analysis(snapshots_dir=snapshots_dir, output_dir=run_dir,
                               experiment_name=name)
        except Exception as e:
            logger.warning(f"TDKPS analysis failed: {e}")

    return run_dir
