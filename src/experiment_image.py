"""
Experiment: Grayscale image compression via Randomized SVD.

Use case where SVD truly compresses storage: storing U[:, :k], s[:k], Vt[:k, :]
costs k * (m + n + 1) floats instead of m * n. The compression ratio is
positive as long as k < m * n / (m + n + 1) (break-even rank).

Outputs:
    data/source_image.npy
    results/image_experiment.json
    report/images/results/image_reconstruction_grid.png
    report/images/results/image_compression_curve.png
    report/images/results/image_singular_spectrum.png
    report/images/results/image_energy_vs_k.png
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project layout.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "report" / "images" / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Make the project root importable so 'src.rsvd' resolves regardless of cwd.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Prefer the project's custom RSVD; fall back to sklearn if unavailable.
RSVD_SOURCE = "custom"
try:
    from src.rsvd import randomized_svd  # type: ignore
except Exception as exc:  # pragma: no cover - fallback path
    warnings.warn(
        f"Custom src.rsvd not available ({exc}); falling back to sklearn.",
        RuntimeWarning,
    )
    from sklearn.utils.extmath import randomized_svd as _sk_rsvd  # type: ignore

    RSVD_SOURCE = "sklearn"

    def randomized_svd(A, k, p=10, q=2, random_state=None):
        return _sk_rsvd(
            A,
            n_components=k,
            n_oversamples=p,
            n_iter=q,
            random_state=random_state,
        )


def load_image() -> tuple[np.ndarray, str]:
    """Return a float64 grayscale image with at least 1e5 pixels."""
    # Priority 1: skimage camera (512x512, no internet).
    try:
        from skimage import data

        img = data.camera().astype(np.float64)
        return img, "skimage.data.camera"
    except Exception as exc:
        warnings.warn(f"skimage.data.camera failed: {exc}", RuntimeWarning)

    # Priority 2: skimage chelsea, convert to grayscale.
    try:
        from skimage import color, data

        img = color.rgb2gray(data.chelsea()).astype(np.float64) * 255.0
        return img, "skimage.data.chelsea (rgb2gray)"
    except Exception as exc:
        warnings.warn(f"skimage.data.chelsea failed: {exc}", RuntimeWarning)

    # Priority 3: synthetic structured 1000x1000 matrix.
    rng = np.random.default_rng(42)
    m, n, r = 1000, 1000, 30
    L = rng.standard_normal((m, r))
    R = rng.standard_normal((r, n))
    img = L @ R + 0.01 * rng.standard_normal((m, n))
    # Re-scale to [0, 255] for plotting parity with images.
    img -= img.min()
    img *= 255.0 / max(img.max(), 1e-12)
    return img, "synthetic_lowrank_1000x1000"


def main() -> None:
    A, source_name = load_image()
    m, n = A.shape
    total = m * n
    if total < 1e5:
        raise RuntimeError(
            f"Image too small ({m}x{n} = {total} elements < 1e5)."
        )

    fro_norm = float(np.linalg.norm(A, "fro"))
    fro_sq = fro_norm**2

    # Save source image as .npy.
    np.save(DATA_DIR / "source_image.npy", A)

    # Break-even rank: largest k with compression ratio >= 1.
    break_even_k = total / (m + n + 1)

    # Candidate k values; clipped to <= min(m, n).
    raw_ks = [5, 10, 20, 50, 100, 200, min(m, n) // 2]
    ks = sorted({int(k) for k in raw_ks if 1 <= int(k) <= min(m, n)})

    results = []
    # Cache U, s, Vt for the largest k so we can build reconstructions for
    # smaller k in the grid plot without redoing work.
    cached = {}
    for k in ks:
        t0 = time.perf_counter()
        U, s, Vt = randomized_svd(A, k=k, p=10, q=2, random_state=0)
        elapsed = time.perf_counter() - t0
        A_k = (U * s) @ Vt
        diff = A - A_k
        fro_err = float(np.linalg.norm(diff, "fro"))
        rel_err = fro_err / fro_norm if fro_norm > 0 else 0.0
        # Energy retention measured against the true ||A||_F^2.
        energy = float(np.sum(s**2) / fro_sq) if fro_sq > 0 else 0.0
        compressed = k * (m + n + 1)
        ratio = total / compressed
        cached[k] = (U, s, Vt, A_k, rel_err)
        results.append(
            {
                "k": int(k),
                "frobenius_error": fro_err,
                "relative_error": rel_err,
                "compression_ratio": float(ratio),
                "compressed_floats": int(compressed),
                "time_seconds": float(elapsed),
                "energy_retention": energy,
            }
        )

    # Use a high-rank RSVD (or full SVD) to estimate the singular spectrum for
    # energy / spectrum diagnostics. We use a single RSVD call at the largest
    # candidate rank we need (capped at min(m, n)).
    spectrum_k = min(m, n)
    # Full SVD is affordable here (image is at most ~512x512 or ~1000x1000).
    U_full, s_full, Vt_full = np.linalg.svd(A, full_matrices=False)
    cum_energy = np.cumsum(s_full**2) / fro_sq

    def smallest_k_for(eta: float) -> int:
        idx = np.searchsorted(cum_energy, eta) + 1
        return int(min(idx, spectrum_k))

    selected_k = {f"{eta:.2f}": smallest_k_for(eta) for eta in (0.90, 0.95, 0.99)}

    output = {
        "image_source": source_name,
        "rsvd_source": RSVD_SOURCE,
        "image_shape": [int(m), int(n)],
        "image_total_elements": int(total),
        "frobenius_norm": fro_norm,
        "break_even_k": float(break_even_k),
        "ks": ks,
        "results": results,
        "selected_k": selected_k,
    }

    out_json = RESULTS_DIR / "image_experiment.json"
    with out_json.open("w") as f:
        json.dump(output, f, indent=2)

    # -------------------- Plots --------------------
    # 1. Reconstruction grid: original + k in {5, 20, 50, 100, 200}.
    grid_targets = [5, 20, 50, 100, 200]
    grid_targets = [k for k in grid_targets if k <= min(m, n)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))
    axes = axes.ravel()
    axes[0].imshow(A, cmap="gray")
    axes[0].set_title(f"Original ({m}x{n})")
    axes[0].axis("off")
    for ax, k in zip(axes[1:], grid_targets):
        if k in cached:
            A_k = cached[k][3]
            rel = cached[k][4]
        else:
            U, s, Vt = randomized_svd(A, k=k, p=10, q=2, random_state=0)
            A_k = (U * s) @ Vt
            rel = float(np.linalg.norm(A - A_k, "fro") / fro_norm)
        ax.imshow(A_k, cmap="gray", vmin=A.min(), vmax=A.max())
        ax.set_title(f"k={k}, rel.err={rel:.3f}")
        ax.axis("off")
    # Hide any remaining empty axes.
    for ax in axes[1 + len(grid_targets):]:
        ax.axis("off")
    fig.suptitle("Randomized SVD reconstructions", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "image_reconstruction_grid.png", dpi=130)
    plt.close(fig)

    # 2. Compression curve.
    ks_arr = np.array(ks, dtype=float)
    ratios = np.array([r["compression_ratio"] for r in results], dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(ks_arr, ratios, "o-", color="C0", label="compression ratio")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="break-even (ratio=1)")
    ax.axvline(break_even_k, color="gray", linestyle=":", linewidth=1,
               label=f"k_be={break_even_k:.1f}")
    # Mark selected k* for each energy threshold.
    colors = {"0.90": "tab:orange", "0.95": "tab:red", "0.99": "tab:purple"}
    for eta_key, k_star in selected_k.items():
        ratio_star = total / (k_star * (m + n + 1))
        ax.scatter([k_star], [ratio_star], color=colors[eta_key], s=70, zorder=5,
                   label=f"k*({eta_key})={k_star}, ratio={ratio_star:.2f}")
    ax.set_xlabel("Rank k")
    ax.set_ylabel("Compression ratio (original / compressed)")
    ax.set_title("Compression ratio vs rank k")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "image_compression_curve.png", dpi=130)
    plt.close(fig)

    # 3. Singular spectrum (log-scale, top 200).
    top = min(200, len(s_full))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.semilogy(np.arange(1, top + 1), s_full[:top], color="C2")
    ax.set_xlabel("Index i")
    ax.set_ylabel("Singular value sigma_i (log scale)")
    ax.set_title(f"Singular spectrum (top {top}) of {source_name}")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "image_singular_spectrum.png", dpi=130)
    plt.close(fig)

    # 4. Cumulative energy.
    fig, ax = plt.subplots(figsize=(7.5, 5))
    xs = np.arange(1, len(cum_energy) + 1)
    ax.plot(xs, cum_energy, color="C3")
    for eta, color in zip((0.90, 0.95, 0.99), ("tab:orange", "tab:red", "tab:purple")):
        ax.axhline(eta, color=color, linestyle="--", linewidth=1,
                   label=f"eta={eta:.2f} -> k*={smallest_k_for(eta)}")
    ax.set_xlabel("Rank k")
    ax.set_ylabel("Cumulative energy sum(sigma_i^2) / ||A||_F^2")
    ax.set_title("Cumulative energy vs rank k")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "image_energy_vs_k.png", dpi=130)
    plt.close(fig)

    # ---- Console summary ----
    best = max(results, key=lambda r: r["compression_ratio"])
    print(f"[image_experiment] source = {source_name}")
    print(f"[image_experiment] rsvd source = {RSVD_SOURCE}")
    print(f"[image_experiment] shape = {m}x{n} (total={total})")
    print(f"[image_experiment] ||A||_F = {fro_norm:.4f}")
    print(f"[image_experiment] break-even k = {break_even_k:.2f}")
    print(f"[image_experiment] selected k (eta=0.90/0.95/0.99) = {selected_k}")
    print("[image_experiment] per-k results:")
    for r in results:
        print(
            f"  k={r['k']:>4d}  rel_err={r['relative_error']:.4f}  "
            f"ratio={r['compression_ratio']:.3f}  "
            f"energy={r['energy_retention']:.4f}  "
            f"time={r['time_seconds']:.3f}s"
        )
    print(
        f"[image_experiment] best compression ratio = {best['compression_ratio']:.3f} at k={best['k']}"
    )
    print(f"[image_experiment] JSON written to {out_json}")


if __name__ == "__main__":
    main()
