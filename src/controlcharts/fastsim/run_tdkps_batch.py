"""Phase B worker with a monkeypatched kernel builder.

maps _generate_kernel_matrix_responses is a 400M-iteration quadruple Python
loop (T*T*A*A) computing Frobenius distances -- ~30min/run. It is exactly a
pairwise Euclidean distance on the flattened (Q*F) response vectors, so one
cdist call reproduces it in seconds. We monkeypatch (no submodule edits).
"""
import sys, glob, json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, "src")
maps_path = Path("maps").resolve()
if str(maps_path) not in sys.path:
    sys.path.insert(0, str(maps_path))

from maps import gmds  # noqa: E402
from sklearn.utils.extmath import randomized_svd  # noqa: E402


def _fast_kernel(Y):
    # Y: (T, A, Q, F). Frobenius(Y[t,i]-Y[t',j]) == Euclidean on flattened rows.
    # BLAS identity d^2 = |a|^2 + |b|^2 - 2 a.b (dgemm) instead of cdist's C
    # double loop. In-place to keep peak memory ~ one n x n matrix.
    T, A, Q, F = Y.shape
    flat = np.ascontiguousarray(Y.reshape(T * A, Q * F), dtype=np.float64)
    sq = np.einsum("ij,ij->i", flat, flat)
    G = flat @ flat.T                    # n x n
    del flat
    G *= -2.0
    G += sq[:, None]
    G += sq[None, :]
    np.maximum(G, 0.0, out=G)            # clip roundoff negatives
    np.sqrt(G, out=G)
    np.fill_diagonal(G, 0.0)
    return G


def _patched_compute_svd(self, kernel_matrix, n_components=None):
    # The stock estimator uses algorithm="full" when n_components==1, which
    # materializes the full n x n U and V (24 min / 18 GB at n=20000). We only
    # need the top component -> randomized top-k SVD, O(n^2 k) and light.
    k = n_components if n_components is not None else self.n_components
    if k is None:
        k = 1
    U, D, Vt = randomized_svd(kernel_matrix, n_components=k, n_iter=7,
                              random_state=self.svd_seed or 0)
    return {'U': U, 'singular_values': D, 'V': Vt.T, 'n_components': len(D)}


gmds._generate_kernel_matrix_responses = _fast_kernel
gmds.TDKPSEstimator._compute_svd = _patched_compute_svd

from controlcharts.tdkps_analysis import run_tdkps_analysis  # noqa: E402

if __name__ == "__main__":
    d = Path(sys.argv[1])
    validate = len(sys.argv) > 2 and sys.argv[2] == "--validate"
    idx = d / "snapshots" / "snapshots_index.json"
    if not idx.exists():
        print(f"SKIP {d.name} (no snapshots_index)"); sys.exit(0)
    name = json.load(open(d / "results_summary.json"))["config"]["experiment"]["name"]
    ref = None
    if validate:
        f = glob.glob(str(d / "*_tdkps_embeddings.npz"))
        if f:
            ref = np.load(f[0])["embedding_matrix"].copy()
    try:
        run_tdkps_analysis(snapshots_dir=d / "snapshots", output_dir=d, experiment_name=name)
        if validate and ref is not None:
            new = np.load(glob.glob(str(d / "*_tdkps_embeddings.npz"))[0])["embedding_matrix"]
            # sign of an SVD component is arbitrary; compare up to per-component sign
            a = ref.reshape(ref.shape[0], -1); b = new.reshape(new.shape[0], -1)
            err = min(np.abs(a - b).max(), np.abs(a + b).max())
            print(f"VALIDATE {d.name}: max|Δ|(sign-free)={err:.3e} shape={new.shape}")
        else:
            print(f"OK {d.name}")
    except Exception as e:
        print(f"FAIL {d.name}: {e}")
