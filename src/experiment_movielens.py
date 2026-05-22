"""
Experiment: Randomized SVD on a sparse MovieLens-like rating matrix.

This script fixes the bug from the original notebook where ``total_energy`` was
computed from the singular values of a *truncated* RSVD instead of from the
matrix itself. The correct Frobenius energy is ``||A||_F^2 = sum_{ij} A_ij^2``,
which for a sparse CSR matrix is obtained via ``A.power(2).sum()``.

The script:
  1. Builds (or loads) a synthetic 5000x2000 sparse rating matrix mimicking
     MovieLens (latent rank-10 structure, ~1% density, ratings in 0.5..5.0).
  2. Computes the EXACT total Frobenius energy from the sparse matrix.
  3. Runs custom randomized SVD (from src.rsvd) at k_max.
  4. Finds the smallest k* achieving cumulative energy ratios 0.90 / 0.95 / 0.99.
  5. Reports Frobenius truncation error and compares storage vs. break-even k.
  6. Saves JSON results and three plots (energy / error / spectrum).

Run from project root:
    python src/experiment_movielens.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless backend, safe on servers
import matplotlib.pyplot as plt
from scipy import sparse

# Make ``src`` importable regardless of CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.rsvd import randomized_svd as _custom_rsvd

    RSVD_BACKEND = "custom"
except Exception as _exc:  # pragma: no cover - fallback path
    warnings.warn(
        f"Custom src.rsvd unavailable ({_exc!r}); falling back to sklearn.",
        RuntimeWarning,
    )
    from sklearn.utils.extmath import randomized_svd as _sk_rsvd

    def _custom_rsvd(A, k, p=10, q=2, random_state=None):
        # sklearn API: returns U, s, Vt with n_oversamples + n_iter knobs.
        U, s, Vt = _sk_rsvd(
            A,
            n_components=k,
            n_oversamples=p,
            n_iter=q,
            random_state=random_state,
        )
        return U, s, Vt

    RSVD_BACKEND = "sklearn"


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "movielens_sparse.npz"  # cache (CSR npz)
ZIP_PATH = DATA_DIR / "ml-25m.zip"             # cache (raw download)
ML_DIR = DATA_DIR / "ml-25m"                   # cache (extracted)
RESULTS_PATH = PROJECT_ROOT / "results" / "movielens_experiment.json"
NUMBERS_TEX = PROJECT_ROOT / "results" / "movielens_numbers.tex"
IMG_DIR = PROJECT_ROOT / "report" / "images" / "results"
PLOT_ENERGY = IMG_DIR / "movielens_energy.png"
PLOT_ERROR = IMG_DIR / "movielens_error.png"
PLOT_SPECTRUM = IMG_DIR / "movielens_spectrum.png"

ML25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
# Bộ dữ liệu này cũng được mirror trên Kaggle:
# https://www.kaggle.com/datasets/grouplens/movielens-25m-dataset

# Bật/tắt synthetic fallback bằng env var MOVIELENS_USE_SYNTHETIC=1
USE_SYNTHETIC_ONLY = os.environ.get("MOVIELENS_USE_SYNTHETIC", "0") == "1"
# Timeout download (giây). Có thể override bằng env var MOVIELENS_DOWNLOAD_TIMEOUT.
DOWNLOAD_TIMEOUT = int(os.environ.get("MOVIELENS_DOWNLOAD_TIMEOUT", "900"))


# --------------------------------------------------------------------------- #
# 1a. Tải MovieLens 25M thật từ GroupLens (= bản Kaggle)
# --------------------------------------------------------------------------- #
def _download_ml25m() -> bool:
    """Tải và giải nén ml-25m.zip vào DATA_DIR/ml-25m/. Trả về True nếu thành công."""
    import urllib.request
    import zipfile
    import shutil

    if (ML_DIR / "ratings.csv").exists():
        return True
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"      Downloading {ML25M_URL} (~250 MB)...")
        try:
            req = urllib.request.Request(
                ML25M_URL,
                headers={"User-Agent": "Mozilla/5.0 (academic project)"},
            )
            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as r:
                with open(ZIP_PATH, "wb") as fp:
                    shutil.copyfileobj(r, fp)
        except Exception as exc:  # network / DNS / timeout
            print(f"      Download FAILED: {exc}")
            if ZIP_PATH.exists():
                ZIP_PATH.unlink()
            return False
    print(f"      Extracting to {ML_DIR}/ ...")
    try:
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(DATA_DIR)
    except Exception as exc:
        print(f"      Extraction FAILED: {exc}")
        return False
    return (ML_DIR / "ratings.csv").exists()


def _build_real_movielens_matrix() -> sparse.csr_matrix | None:
    """Đọc ratings.csv thật, build CSR. Trả về None nếu tải thất bại."""
    if not _download_ml25m():
        return None
    import pandas as pd

    csv_path = ML_DIR / "ratings.csv"
    print(f"      Reading {csv_path.name} ...")
    df = pd.read_csv(csv_path, usecols=["userId", "movieId", "rating"])
    print(f"      Loaded {len(df):,} ratings")

    # Ánh xạ ID liên tục
    uniq_u = df["userId"].unique()
    uniq_m = df["movieId"].unique()
    u_map = {u: i for i, u in enumerate(uniq_u)}
    m_map = {v: j for j, v in enumerate(uniq_m)}
    rows = df["userId"].map(u_map).to_numpy(dtype=np.int32)
    cols = df["movieId"].map(m_map).to_numpy(dtype=np.int32)
    vals = df["rating"].to_numpy(dtype=np.float64)
    A = sparse.csr_matrix(
        (vals, (rows, cols)), shape=(len(uniq_u), len(uniq_m))
    )
    print(f"      Built CSR shape={A.shape}, nnz={A.nnz:,}")
    return A


# --------------------------------------------------------------------------- #
# 1b. Synthetic fallback (mô phỏng MovieLens khi không tải được)
# --------------------------------------------------------------------------- #
def _build_synthetic_matrix(
    m: int = 5000,
    n: int = 2000,
    latent_rank: int = 10,
    density: float = 0.01,
    seed: int = 42,
) -> sparse.csr_matrix:
    """Return a sparse CSR rating matrix shaped (m, n).

    Cached at :data:`DATA_PATH`. Ratings are clipped to the MovieLens grid
    ``{0.5, 1.0, ..., 5.0}``; ~``1 - density`` of the entries are masked out
    to mimic real-world sparsity.

    Latent structure is injected in two ways so the resulting sparse matrix
    still exhibits a decaying singular spectrum (as real MovieLens does):
      1. A rank-``latent_rank`` user/item factor model produces the rating
         magnitudes.
      2. Per-user and per-item popularity biases skew the sampling
         probability, so heavy raters and popular movies dominate the
         observed mass — this yields strong leading singular values.
    """
    rng = np.random.default_rng(seed)

    # --- Latent factors (rank-r model) ----------------------------------
    # Scaling chosen so that U V^T + global_mean lives mostly in [0.5, 5.0].
    U_true = rng.standard_normal(size=(m, latent_rank)) * 0.35
    V_true = rng.standard_normal(size=(n, latent_rank)) * 0.35

    # --- Popularity biases ---------------------------------------------
    # Strongly skewed (log-normal) user activity and movie popularity.
    user_pop = rng.lognormal(mean=0.0, sigma=1.0, size=m)
    item_pop = rng.lognormal(mean=0.0, sigma=1.0, size=n)
    sample_p = np.outer(user_pop, item_pop)
    sample_p /= sample_p.sum()  # normalize to a discrete distribution

    nnz_target = int(round(m * n * density))

    # Sample (row, col) flat indices WITH replacement under the biased
    # distribution, then keep unique pairs only. Oversample to land near
    # the target nnz after deduplication.
    over = 1.6
    n_draws = min(int(nnz_target * over), m * n)
    flat = rng.choice(m * n, size=n_draws, replace=True, p=sample_p.ravel())
    flat = np.unique(flat)
    if flat.size > nnz_target:
        flat = rng.choice(flat, size=nnz_target, replace=False)
    rows = flat // n
    cols = flat % n

    # --- Rating values --------------------------------------------------
    vals = np.einsum("ij,ij->i", U_true[rows], V_true[cols])
    # Center near a 3.5 global mean (typical for MovieLens).
    vals = vals + 3.5
    # Snap to half-star grid and clip.
    vals = np.clip(np.round(vals * 2.0) / 2.0, 0.5, 5.0)

    keep = vals > 0
    rows, cols, vals = rows[keep], cols[keep], vals[keep]

    A = sparse.coo_matrix((vals, (rows, cols)), shape=(m, n)).tocsr()
    A.sum_duplicates()
    A.eliminate_zeros()
    return A


# --------------------------------------------------------------------------- #
# 1c. Hàm chính: ưu tiên dữ liệu thật, fallback synthetic
# --------------------------------------------------------------------------- #
def build_or_load_matrix() -> tuple[sparse.csr_matrix, str]:
    """Trả về (CSR ma trận đánh giá, data_source).

    Thứ tự ưu tiên:
      1. Cache CSR đã lưu (data/movielens_sparse.npz) -- nhãn tự lưu trong cache.
      2. Tải MovieLens 25M thật từ GroupLens.
      3. Sinh synthetic (env var MOVIELENS_USE_SYNTHETIC=1 buộc dùng phương án này).
    """
    # 1) Cache npz
    label_file = DATA_DIR / "movielens_source.txt"
    if DATA_PATH.exists() and label_file.exists():
        source = label_file.read_text().strip()
        print(f"      Cache hit: {DATA_PATH.name} (source={source})")
        return sparse.load_npz(DATA_PATH).tocsr(), source

    # 2) Tải dữ liệu thật trừ khi bị disable
    if not USE_SYNTHETIC_ONLY:
        print("      Trying real MovieLens 25M (GroupLens / Kaggle mirror)...")
        A = _build_real_movielens_matrix()
        if A is not None:
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            sparse.save_npz(DATA_PATH, A)
            label_file.write_text("real_grouplens")
            return A, "real_grouplens"
        print("      Real data unavailable — falling back to synthetic.")

    # 3) Synthetic fallback
    print("      Building synthetic 5000x2000 sparse matrix...")
    A = _build_synthetic_matrix()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(DATA_PATH, A)
    label_file.write_text("synthetic_fallback")
    return A, "synthetic_fallback"


# --------------------------------------------------------------------------- #
# 2. Energy / error helpers
# --------------------------------------------------------------------------- #
def find_k_star(cum_ratio: np.ndarray, eta: float) -> int | None:
    """Smallest 1-indexed k such that ``cum_ratio[k-1] >= eta`` (or None)."""
    mask = cum_ratio >= eta
    if not mask.any():
        return None
    return int(np.argmax(mask)) + 1


# --------------------------------------------------------------------------- #
# 3. Plotting
# --------------------------------------------------------------------------- #
def plot_energy(cum_ratio: np.ndarray, selected_k: dict, path: Path) -> None:
    ks = np.arange(1, len(cum_ratio) + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ks, cum_ratio, color="#1f77b4", linewidth=1.5,
            label="Cumulative energy ratio")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)

    colors = {"0.90": "#2ca02c", "0.95": "#ff7f0e", "0.99": "#d62728"}
    for eta_str, k_star in selected_k.items():
        eta = float(eta_str)
        ax.axhline(eta, color=colors[eta_str], linestyle="--",
                   linewidth=1.0, alpha=0.7)
        if k_star is not None:
            ax.axvline(k_star, color=colors[eta_str], linestyle=":",
                       linewidth=0.9, alpha=0.6)
            ax.annotate(
                f"$\\eta={eta:.2f}$, $k^*={k_star}$",
                xy=(k_star, eta),
                xytext=(k_star + 5, eta - 0.04),
                fontsize=9,
                color=colors[eta_str],
            )
        else:
            ax.text(
                len(cum_ratio) * 0.6,
                eta + 0.005,
                f"$\\eta={eta:.2f}$ not reached at $k_{{\\max}}$",
                fontsize=9,
                color=colors[eta_str],
            )

    ax.set_xlabel("Rank $k$")
    ax.set_ylabel(r"$\sum_{i \leq k}\sigma_i^2 / \|A\|_F^2$")
    ax.set_title("MovieLens (sparse synthetic): cumulative energy vs rank")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error(errors: np.ndarray, path: Path) -> None:
    ks = np.arange(1, len(errors) + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.semilogy(ks, errors, color="#d62728", linewidth=1.5)
    ax.set_xlabel("Rank $k$")
    ax.set_ylabel(r"$\|A - A_k\|_F$ (log scale)")
    ax.set_title("MovieLens (sparse synthetic): Frobenius truncation error")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spectrum(s: np.ndarray, path: Path, top: int = 100) -> None:
    top = min(top, len(s))
    idx = np.arange(1, top + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.semilogy(idx, s[:top], marker="o", markersize=3, linewidth=1.0,
                color="#1f77b4")
    ax.set_xlabel("Index $i$")
    ax.set_ylabel(r"$\sigma_i$ (log scale)")
    ax.set_title(f"MovieLens (sparse synthetic): top-{top} singular values")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. Main
# --------------------------------------------------------------------------- #
def main() -> None:
    np.random.seed(42)

    print("[1/6] Loading MovieLens rating matrix (real first, synthetic fallback)...")
    A, data_source = build_or_load_matrix()
    m, n = A.shape
    nnz = int(A.nnz)
    density = nnz / (m * n)
    print(f"      shape = {A.shape}, nnz = {nnz}, density = {density:.4%}")

    # ---- BUG FIX: total energy must come from the matrix itself, NOT from
    # the squared singular values of a truncated RSVD (those would only sum
    # to <= true energy, biasing all ratios upward). ----
    print("[2/6] Computing exact Frobenius energy from A...")
    total_energy = float(A.power(2).sum())
    frob_norm = float(np.sqrt(total_energy))
    print(f"      ||A||_F^2 = {total_energy:.6e}, ||A||_F = {frob_norm:.6e}")

    # k_max must be large enough that the cumulative-energy curve crosses the
    # high-eta targets; for a sparse rating-style matrix with heavy popularity
    # bias the spectrum decays slowly, so k_max needs to be well above the
    # break-even threshold k_be = nnz / (m + n + 1). We choose 1500 < n so
    # that RSVD remains well-posed.
    k_max = 1500
    print(f"[3/6] Running randomized SVD (backend={RSVD_BACKEND}) at k_max={k_max}...")
    t0 = time.perf_counter()
    U, s, Vt = _custom_rsvd(A, k=k_max, p=10, q=2, random_state=42)
    rsvd_seconds = time.perf_counter() - t0
    print(f"      done in {rsvd_seconds:.2f} s; s.shape = {s.shape}")

    # Cumulative singular-value energy. Note: because RSVD is approximate,
    # cum_energy[k_max-1] may be slightly less than total_energy (it is a
    # *lower bound* — sum of squared SVs of a rank-k_max approximation).
    cum_energy = np.cumsum(s ** 2)
    cum_ratio = cum_energy / total_energy

    print("[4/6] Locating k* for each target energy ratio...")
    etas = [0.90, 0.95, 0.99]
    selected_k: dict[str, int | None] = {}
    for eta in etas:
        k_star = find_k_star(cum_ratio, eta)
        selected_k[f"{eta:.2f}"] = k_star
        if k_star is None:
            print(f"      eta={eta:.2f}: not reached at k_max={k_max} "
                  f"(max ratio = {cum_ratio[-1]:.4f})")
        else:
            print(f"      eta={eta:.2f}: k* = {k_star}")

    # Frobenius error for each k via energy identity. Clamp negatives that
    # arise from RSVD approximation noise.
    frob_errors = np.sqrt(np.clip(total_energy - cum_energy, 0.0, None))

    # Storage analysis.
    original_storage = nnz * 16  # CSR: 8B data (float64) + 8B col_idx
    break_even_k = nnz / (m + n + 1)
    print("[5/6] Storage analysis...")
    print(f"      original CSR storage ~= {original_storage:,} B "
          f"(nnz*16)")
    print(f"      break-even k_be = nnz / (m+n+1) = {break_even_k:.2f}")

    results_per_eta = []
    for eta in etas:
        eta_key = f"{eta:.2f}"
        k_star = selected_k[eta_key]
        if k_star is None:
            results_per_eta.append({
                "eta": eta,
                "k": None,
                "energy_ratio": float(cum_ratio[-1]),
                "frobenius_error": float(frob_errors[-1]),
                "compression_storage_bytes": None,
                "original_storage_bytes": int(original_storage),
                "compresses": False,
                "note": f"eta not reached at k_max={k_max}",
            })
            continue
        compressed_storage = int(k_star * (m + n + 1) * 8)
        compresses = compressed_storage < original_storage
        results_per_eta.append({
            "eta": eta,
            "k": int(k_star),
            "energy_ratio": float(cum_ratio[k_star - 1]),
            "frobenius_error": float(frob_errors[k_star - 1]),
            "compression_storage_bytes": compressed_storage,
            "original_storage_bytes": int(original_storage),
            "compression_ratio": compressed_storage / original_storage,
            "compresses": bool(compresses),
        })
        print(
            f"      eta={eta:.2f}: k*={k_star}, "
            f"err={frob_errors[k_star - 1]:.4e}, "
            f"compressed={compressed_storage:,} B "
            f"({compressed_storage / original_storage:.2f}x original)"
        )

    # Did any selected_k beat the break-even? Build the conclusion.
    any_compress = any(r.get("compresses") for r in results_per_eta)
    if any_compress:
        conclusion = (
            "Compression succeeds for at least one target eta because k* < k_be."
        )
    else:
        conclusion = (
            f"Compression fails: every k* >= break-even k_be ~= "
            f"{break_even_k:.1f}, so a dense rank-k factorization needs more "
            f"storage than the original CSR matrix."
        )
    print(f"      conclusion: {conclusion}")

    print("[6/6] Writing results and plots...")
    out = {
        "data_source": data_source,
        "shape": [m, n],
        "nnz": nnz,
        "density": density,
        "total_energy": total_energy,
        "frobenius_norm": frob_norm,
        "k_max": k_max,
        "rsvd_backend": RSVD_BACKEND,
        "rsvd_seconds": rsvd_seconds,
        "break_even_k": break_even_k,
        "selected_k": {key: (int(v) if v is not None else None)
                       for key, v in selected_k.items()},
        "results": results_per_eta,
        "conclusion": conclusion,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"      JSON -> {RESULTS_PATH}")

    # ---- LaTeX macros: report đọc trực tiếp các giá trị này ----
    src_label = {
        "real_grouplens": "MovieLens~25M (GroupLens / Kaggle)",
        "synthetic_fallback": "MovieLens-like synthetic ($5000\\times 2000$)",
    }.get(data_source, data_source)

    def _fmt_int(x: int) -> str:
        return f"{x:,}".replace(",", "{,}")

    k90 = selected_k["0.90"]
    k95 = selected_k["0.95"]
    k99 = selected_k["0.99"]
    err_k95 = next(
        (r["frobenius_error"] for r in results_per_eta
         if r.get("eta") == 0.95 and r.get("k") is not None),
        frob_errors[-1],
    )

    tex_lines = [
        "% Auto-generated by src/experiment_movielens.py -- do not edit.",
        f"\\renewcommand{{\\mlSource}}{{{src_label}}}",
        f"\\renewcommand{{\\mlM}}{{{_fmt_int(m)}}}",
        f"\\renewcommand{{\\mlN}}{{{_fmt_int(n)}}}",
        f"\\renewcommand{{\\mlNNZ}}{{{_fmt_int(nnz)}}}",
        f"\\renewcommand{{\\mlDensity}}{{{density*100:.2f}\\%}}",
        f"\\renewcommand{{\\mlEnergy}}{{$\\sim {total_energy:.2e}$}}",
        f"\\renewcommand{{\\mlKbe}}{{{break_even_k:.1f}}}",
        f"\\renewcommand{{\\mlKstarNinety}}"
            f"{{{'---' if k90 is None else _fmt_int(k90)}}}",
        f"\\renewcommand{{\\mlKstarNinetyFive}}"
            f"{{{'---' if k95 is None else _fmt_int(k95)}}}",
        f"\\renewcommand{{\\mlKstarNinetyNine}}"
            f"{{{'---' if k99 is None else _fmt_int(k99)}}}",
        f"\\renewcommand{{\\mlFroberror}}{{{_fmt_int(int(round(err_k95)))}}}",
        "",
    ]
    NUMBERS_TEX.parent.mkdir(parents=True, exist_ok=True)
    NUMBERS_TEX.write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"      TeX  -> {NUMBERS_TEX}")

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    plot_energy(cum_ratio, selected_k, PLOT_ENERGY)
    plot_error(frob_errors, PLOT_ERROR)
    plot_spectrum(s, PLOT_SPECTRUM, top=100)
    print(f"      Plots -> {PLOT_ENERGY.name}, {PLOT_ERROR.name}, "
          f"{PLOT_SPECTRUM.name}")


if __name__ == "__main__":
    main()
