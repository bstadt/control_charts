"""Precompute the response-alphabet embedding cache ONCE (one model load).

The no-LLM response alphabet is finite: every NQ answer, the temporal-value
integer strings, the quine, and "I don't know". Embedding them all once and
caching to disk lets a parallel sweep run with zero per-cell model loads
(the loads were OOMing the tower at parallelism).

Usage:
  python -m controlcharts.fastsim.precompute_embed_cache \
      --data-path /home/bstadt/root/data/nq_embedded.parquet \
      --out /tmp/fastsim_embed_cache.pkl
Then export FASTSIM_EMBED_CACHE=/tmp/fastsim_embed_cache.pkl before the sweep.
"""
import argparse
import logging
import pickle

import numpy as np
import pandas as pd

from .core import QUINE_TEXT, IDK_TEXT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                    help="seeds the sweep will use (determines which answers are sampled)")
    ap.add_argument("--total-questions", type=int, default=50)
    ap.add_argument("--n-temporal", type=int, default=10)
    ap.add_argument("--max-temporal-int", type=int, default=400,
                    help="cache str(0..N) for temporal values")
    args = ap.parse_args()

    # Replicate the runner's per-seed question sampling so we embed exactly the
    # answers that can ever appear -- a few hundred strings, not all of NQ.
    df = pd.read_parquet(args.data_path, columns=["answer"])
    all_answers = df["answer"].astype(str).tolist()
    n_nt = min(args.total_questions - args.n_temporal, len(all_answers))
    used = set()
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(all_answers), size=n_nt, replace=False)
        used.update(all_answers[i] for i in idx)
    strings = sorted(used)
    strings += [QUINE_TEXT, IDK_TEXT]
    strings += [str(i) for i in range(args.max_temporal_int + 1)]
    strings = sorted(set(strings))
    logger.info(f"Embedding {len(strings)} response strings for seeds {args.seeds} once")

    from sentence_transformers import SentenceTransformer
    from ..embedding import MODEL_ID, MODEL_REVISION
    model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True)
    embs = model.encode(strings, batch_size=128, show_progress_bar=True,
                        normalize_embeddings=False, convert_to_numpy=True).astype(np.float32)
    cache = {s: embs[i] for i, s in enumerate(strings)}
    with open(args.out, "wb") as f:
        pickle.dump(cache, f, protocol=4)
    logger.info(f"Wrote cache: {len(cache)} strings -> {args.out}")


if __name__ == "__main__":
    main()
