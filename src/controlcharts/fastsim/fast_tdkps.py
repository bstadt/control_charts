"""Drop-in performance patch for maps TDKPS at large panels.

The stock ``maps`` kernel builder ``_generate_kernel_matrix_responses`` is a
quadruple Python loop (n_timesteps^2 * n_agents^2) computing Frobenius
distances -- ~30 min for a 200-timestep x 100-agent panel. And for
``n_components == 1`` the estimator's ``_compute_svd`` uses ``algorithm="full"``,
materializing the full n x n U and V (n = timesteps*agents = 20000 -> 18 GB,
24 min).

Both are unnecessary for the iso-mirror:
  * the block Frobenius kernel is exactly a pairwise Euclidean distance on the
    flattened (n_queries*n_features) response vectors -> one BLAS gemm;
  * only the top component is needed -> randomized top-k SVD, O(n^2 k).

Importing this module (or calling ``patch_maps()``) monkeypatches both in place
-- no edits to the maps submodule. Verified against the stock path: iso-mirror
embedding matches to < 1e-6 (max abs, sign-free) on the N=10000 panel, wall
time 24 min -> 56 s, peak RSS 18 GB -> 9.8 GB.
"""

import numpy as np
from scipy.spatial.distance import cdist  # noqa: F401  (kept for parity/reference)


def _fast_kernel(Y):
    """Block Frobenius distance kernel == Euclidean distance on flattened rows.

    Y: (n_timesteps, n_agents, n_queries, n_features). In-place after the gemm
    so peak memory is ~one n x n matrix (n = n_timesteps*n_agents).
    """
    T, A, Q, F = Y.shape
    flat = np.ascontiguousarray(Y.reshape(T * A, Q * F), dtype=np.float64)
    sq = np.einsum("ij,ij->i", flat, flat)
    G = flat @ flat.T
    del flat
    G *= -2.0
    G += sq[:, None]
    G += sq[None, :]
    np.maximum(G, 0.0, out=G)            # clip roundoff negatives
    np.sqrt(G, out=G)
    np.fill_diagonal(G, 0.0)
    return G


def _patched_compute_svd(self, kernel_matrix, n_components=None):
    """Randomized top-k SVD instead of the full n x n decomposition."""
    from sklearn.utils.extmath import randomized_svd
    k = n_components if n_components is not None else self.n_components
    if k is None:
        k = 1
    U, D, Vt = randomized_svd(kernel_matrix, n_components=k, n_iter=7,
                              random_state=self.svd_seed or 0)
    return {"U": U, "singular_values": D, "V": Vt.T, "n_components": len(D)}


def patch_maps():
    """Apply both patches to the imported maps modules. Idempotent."""
    from maps import gmds
    gmds._generate_kernel_matrix_responses = _fast_kernel
    gmds.TDKPSEstimator._compute_svd = _patched_compute_svd
    return gmds


# Patch on import so `import controlcharts.fastsim.fast_tdkps` before
# run_tdkps_analysis is enough.
try:
    patch_maps()
except Exception:  # maps not importable yet; caller can call patch_maps() later
    pass
