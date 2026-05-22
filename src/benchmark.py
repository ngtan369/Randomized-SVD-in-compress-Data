"""
Benchmark hiệu năng và độ chính xác của các phương pháp SVD top-k.

So sánh ba phương pháp:
  1. custom_rsvd     -> src.rsvd.randomized_svd (Halko-Martinsson-Tropp).
  2. sklearn_rsvd    -> sklearn.utils.extmath.randomized_svd (baseline).
  3. numpy_full_svd  -> numpy.linalg.svd đầy đủ, sau đó cắt top-k.

Ma trận test được sinh dense với phổ suy giảm hình học decay[i] = 0.95^i,
cộng nhiễu Gaussian biên độ nhỏ. Ghi kết quả ra results/benchmark.json và
vẽ hai biểu đồ runtime, accuracy trong report/images/results/.

Chạy:
    cd /path/to/Randomized-SVD-in-compress-Data
    python src/benchmark.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Đảm bảo import được src.rsvd dù chạy từ thư mục dự án hay khác.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Method 1: custom RSVD (có fallback nếu chưa tồn tại) ----------------
try:
    from src.rsvd import randomized_svd as custom_randomized_svd  # type: ignore

    HAS_CUSTOM_RSVD = True
except Exception as exc:  # pragma: no cover - chỉ chạy khi Agent 1 chưa xong
    print(
        f"[CẢNH BÁO] Không import được src.rsvd: {exc!r}. "
        "Bỏ qua method 'custom_rsvd'."
    )
    HAS_CUSTOM_RSVD = False
    custom_randomized_svd = None  # type: ignore

# --- Method 2: sklearn RSVD ----------------------------------------------
from sklearn.utils.extmath import randomized_svd as sklearn_randomized_svd


# -------------------- cấu hình thí nghiệm -----------------------------------
SIZES = [200, 500, 1000, 2000]
K = 50
P = 10
Q = 2
N_RUNS = 3
SEED = 42
FULL_SVD_TIMEOUT_SEC = 60.0  # nếu numpy full SVD vượt ngưỡng -> đánh dấu timeout

RESULTS_JSON = PROJECT_ROOT / "results" / "benchmark.json"
RUNTIME_PNG = PROJECT_ROOT / "report" / "images" / "results" / "benchmark_runtime.png"
ACCURACY_PNG = PROJECT_ROOT / "report" / "images" / "results" / "benchmark_accuracy.png"


# -------------------- sinh ma trận test -------------------------------------
def make_test_matrix(n: int, seed: int = SEED) -> np.ndarray:
    """Sinh ma trận vuông n x n với phổ suy giảm hình học 0.95^i + nhiễu nhỏ.

    Cấu trúc: A = U_rand @ diag(decay) @ V_rand.T + noise, ở đó U_rand, V_rand
    là các ma trận trực giao thu từ QR của Gaussian ngẫu nhiên. Phổ suy giảm
    đặc trưng cho nhiều bộ dữ liệu thực, giúp đánh giá đúng lợi thế của RSVD.
    """
    rng = np.random.default_rng(seed)

    G1 = rng.standard_normal(size=(n, n))
    G2 = rng.standard_normal(size=(n, n))
    U_rand, _ = np.linalg.qr(G1)
    V_rand, _ = np.linalg.qr(G2)

    decay = np.power(0.95, np.arange(n))  # 1, 0.95, 0.95^2, ...
    A = (U_rand * decay) @ V_rand.T  # tương đương U_rand @ diag(decay) @ V_rand.T

    noise = 1e-6 * rng.standard_normal(size=(n, n))
    return A + noise


# -------------------- các adapter chung -------------------------------------
def run_custom_rsvd(A, k):
    return custom_randomized_svd(A, k=k, p=P, q=Q, random_state=SEED)


def run_sklearn_rsvd(A, k):
    U, s, Vt = sklearn_randomized_svd(
        A,
        n_components=k,
        n_oversamples=P,
        n_iter=Q,
        random_state=SEED,
    )
    return U, s, Vt


def run_numpy_full(A, k):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k], s[:k], Vt[:k, :]


METHODS = []
if HAS_CUSTOM_RSVD:
    METHODS.append(("custom_rsvd", run_custom_rsvd))
METHODS.append(("sklearn_rsvd", run_sklearn_rsvd))
METHODS.append(("numpy_full_svd", run_numpy_full))


# -------------------- đo lường ----------------------------------------------
def frobenius_error(A: np.ndarray, U: np.ndarray, s: np.ndarray, Vt: np.ndarray) -> float:
    """||A - U diag(s) Vt||_F."""
    A_hat = (U * s) @ Vt
    return float(np.linalg.norm(A - A_hat, ord="fro"))


def time_call(fn, A, k, n_runs: int) -> tuple[float, tuple]:
    """Chạy fn(A, k) n_runs lần, trả về (min_time, last_result)."""
    best = float("inf")
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(A, k)
        elapsed = time.perf_counter() - t0
        if elapsed < best:
            best = elapsed
    return best, result


# -------------------- vòng lặp benchmark ------------------------------------
def run_benchmark():
    results = []
    for n in SIZES:
        print(f"\n=== Ma trận {n} x {n} ===")
        A = make_test_matrix(n)
        norm_A = float(np.linalg.norm(A, ord="fro"))
        print(f"  ||A||_F = {norm_A:.4f}")

        for method_name, fn in METHODS:
            # Với size 2000 + numpy full SVD: chạy 1 lần và áp dụng timeout mềm.
            if n >= 2000 and method_name == "numpy_full_svd":
                t0 = time.perf_counter()
                U, s, Vt = fn(A, K)
                elapsed = time.perf_counter() - t0
                timed_out = elapsed > FULL_SVD_TIMEOUT_SEC
                runs_used = 1
            else:
                elapsed, (U, s, Vt) = time_call(fn, A, K, N_RUNS)
                timed_out = False
                runs_used = N_RUNS

            err = frobenius_error(A, U, s, Vt)
            rel_err = err / norm_A if norm_A > 0 else float("nan")

            print(
                f"  [{method_name:>14s}] time={elapsed:8.4f}s  "
                f"||A-A_k||_F={err:.4e}  rel={rel_err:.4e}"
                + (f"  (runs={runs_used}, timeout!)" if timed_out else f"  (runs={runs_used})")
            )

            results.append(
                {
                    "size": n,
                    "method": method_name,
                    "time_sec": elapsed,
                    "frobenius_error": err,
                    "relative_error": rel_err,
                    "n_runs_used": runs_used,
                    "timed_out": timed_out,
                }
            )

    payload = {
        "config": {
            "k": K,
            "p": P,
            "q": Q,
            "n_runs": N_RUNS,
            "seed": SEED,
            "matrix_structure": "U_rand @ diag(0.95^i) @ V_rand.T + 1e-6 * noise",
            "full_svd_timeout_sec": FULL_SVD_TIMEOUT_SEC,
        },
        "sizes": SIZES,
        "methods": [m for m, _ in METHODS],
        "results": results,
    }

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nĐã ghi {RESULTS_JSON}")
    return payload


# -------------------- vẽ biểu đồ --------------------------------------------
METHOD_LABELS = {
    "custom_rsvd": "RSVD tự cài (HMT)",
    "sklearn_rsvd": "sklearn randomized_svd",
    "numpy_full_svd": "numpy full SVD (top-k)",
}

METHOD_STYLES = {
    "custom_rsvd": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "sklearn_rsvd": {"color": "#2ca02c", "marker": "s", "linestyle": "--"},
    "numpy_full_svd": {"color": "#d62728", "marker": "^", "linestyle": "-."},
}


def _group_by_method(results, methods, sizes, field):
    """Trả về dict {method: [values theo thứ tự sizes]}."""
    table = {m: [None] * len(sizes) for m in methods}
    size_idx = {s: i for i, s in enumerate(sizes)}
    for row in results:
        table[row["method"]][size_idx[row["size"]]] = row[field]
    return table


def plot_runtime(payload):
    sizes = payload["sizes"]
    methods = payload["methods"]
    times = _group_by_method(payload["results"], methods, sizes, "time_sec")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m in methods:
        ys = times[m]
        style = METHOD_STYLES.get(m, {})
        ax.plot(sizes, ys, label=METHOD_LABELS.get(m, m), markersize=8, linewidth=2, **style)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Kích thước ma trận n (n x n)")
    ax.set_ylabel("Thời gian chạy (giây)")
    ax.set_title("Thời gian chạy SVD top-k (k=50)")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    RUNTIME_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(RUNTIME_PNG, dpi=150)
    plt.close(fig)
    print(f"Đã ghi {RUNTIME_PNG}")


def plot_accuracy(payload):
    sizes = payload["sizes"]
    methods = payload["methods"]
    rels = _group_by_method(payload["results"], methods, sizes, "relative_error")

    x = np.arange(len(sizes))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, m in enumerate(methods):
        ys = rels[m]
        style = METHOD_STYLES.get(m, {})
        ax.bar(
            x + (i - (len(methods) - 1) / 2) * width,
            ys,
            width=width,
            label=METHOD_LABELS.get(m, m),
            color=style.get("color"),
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Kích thước ma trận n (n x n)")
    ax.set_ylabel("Sai số tương đối ||A - A_k||_F / ||A||_F")
    ax.set_title("Sai số tương đối ||A-A_k||_F/||A||_F")
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    ACCURACY_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ACCURACY_PNG, dpi=150)
    plt.close(fig)
    print(f"Đã ghi {ACCURACY_PNG}")


# -------------------- entry point ------------------------------------------
def main():
    payload = run_benchmark()
    plot_runtime(payload)
    plot_accuracy(payload)


if __name__ == "__main__":
    main()
