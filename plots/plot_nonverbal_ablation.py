

import json
import os
from typing import Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Config
# ---------------------------
BOOTSTRAP_N = 5000   # 10k also fine for tighter CI
DPI = 300
OUTDIR_NAME = "figs_token_removal"

# Common PHQ-8 severity buckets
BANDS = [(0, 4), (5, 9), (10, 14), (15, None)]

# Legend labels (as requested)
LABEL_RETAINED = "Nonverbal annotations retained"
LABEL_REMOVED = "Nonverbal annotations removed"


# ---------------------------
# Robust JSON parsing
# ---------------------------
def _try_parse_pair(item: Any) -> Optional[Tuple[float, float]]:
    # Case 1: [true, pred]
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        try:
            return float(item[0]), float(item[1])
        except Exception:
            return None

    # Case 2: dict with common keys
    if isinstance(item, dict):
        key_pairs = [
            ("y_true", "y_pred"),
            ("true", "pred"),
            ("label", "pred"),
            ("gt", "pred"),
            ("phq", "pred"),
        ]
        for kt, kp in key_pairs:
            if kt in item and kp in item:
                try:
                    return float(item[kt]), float(item[kp])
                except Exception:
                    return None
    return None


def load_predictions(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and "predictions" in obj:
        raw = obj["predictions"]
    elif isinstance(obj, list):
        raw = obj
    else:
        raise ValueError(f"[ERROR] Unrecognized JSON structure: {path}")

    if not isinstance(raw, list):
        raise ValueError(f"[ERROR] 'predictions' is not a list: {path}")

    pairs: List[Tuple[float, float]] = []
    for it in raw:
        p = _try_parse_pair(it)
        if p is not None:
            pairs.append(p)

    if len(pairs) == 0:
        raise ValueError(
            f"[ERROR] Cannot parse (y_true, y_pred) pairs from {path}. "
            f"Expected {{'predictions': [[true,pred],...]}} or list-of-pairs."
        )

    y_true = np.array([p[0] for p in pairs], dtype=float)
    y_pred = np.array([p[1] for p in pairs], dtype=float)
    return y_true, y_pred


# ---------------------------
# Metrics + Bootstrap CI
# ---------------------------
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # pure numpy version (no p-value)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = metric_fn(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------
# Plot helpers
# ---------------------------
def ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def save_fig(fig, outdir: str, name: str) -> None:
    png = os.path.join(outdir, f"{name}.png")
    pdf = os.path.join(outdir, f"{name}.pdf")
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def diagonal_limits(*arrays: np.ndarray) -> Tuple[float, float]:
    mn = float(min(np.min(a) for a in arrays))
    mx = float(max(np.max(a) for a in arrays))
    pad = 0.05 * (mx - mn + 1e-9)
    return mn - pad, mx + pad


# ---------------------------
# Figures
# ---------------------------
def plot_bar_metrics_ci(dev_retained, dev_removed, test_retained, test_removed, outdir: str) -> None:
    """
    Grouped bars for Dev/Test MAE/RMSE + bootstrap 95% CI.
    """
    (ytdrtn, ypdrtn) = dev_retained
    (ytdrmv, ypdrmv) = dev_removed
    (ytt_rtn, ypt_rtn) = test_retained
    (ytt_rmv, ypt_rmv) = test_removed

    labels = ["Dev MAE", "Dev RMSE", "Test MAE", "Test RMSE"]

    vals_retained = [
        mae(ytdrtn, ypdrtn), rmse(ytdrtn, ypdrtn),
        mae(ytt_rtn, ypt_rtn), rmse(ytt_rtn, ypt_rtn)
    ]
    vals_removed = [
        mae(ytdrmv, ypdrmv), rmse(ytdrmv, ypdrmv),
        mae(ytt_rmv, ypt_rmv), rmse(ytt_rmv, ypt_rmv)
    ]

    ci_retained = [
        bootstrap_ci(ytdrtn, ypdrtn, mae,  seed=1),
        bootstrap_ci(ytdrtn, ypdrtn, rmse, seed=2),
        bootstrap_ci(ytt_rtn, ypt_rtn, mae,  seed=3),
        bootstrap_ci(ytt_rtn, ypt_rtn, rmse, seed=4),
    ]
    ci_removed = [
        bootstrap_ci(ytdrmv, ypdrmv, mae,  seed=11),
        bootstrap_ci(ytdrmv, ypdrmv, rmse, seed=12),
        bootstrap_ci(ytt_rmv, ypt_rmv, mae,  seed=13),
        bootstrap_ci(ytt_rmv, ypt_rmv, rmse, seed=14),
    ]

    err_retained = np.array([[v - lo, hi - v] for v, (lo, hi) in zip(vals_retained, ci_retained)]).T
    err_removed = np.array([[v - lo, hi - v] for v, (lo, hi) in zip(vals_removed, ci_removed)]).T

    x = np.arange(len(labels))
    width = 0.36

    fig = plt.figure(figsize=(10.5, 5.2))
    ax = fig.add_subplot(111)

    b1 = ax.bar(x - width/2, vals_retained, width, yerr=err_retained, capsize=4, label=LABEL_RETAINED)
    b2 = ax.bar(x + width/2, vals_removed, width, yerr=err_removed, capsize=4, label=LABEL_REMOVED)

    ax.set_ylabel("Error (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Effect of removing nonverbal annotations on PHQ-8 regression")
    ax.legend(frameon=False)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width()/2, h),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)

    fig.tight_layout()
    save_fig(fig, outdir, "fig_bar_metrics_ci")


def plot_paired_absdiff(dev_retained, dev_removed, test_retained, test_removed, outdir: str) -> None:
    """
    Boxplot of per-subject abs error difference:
      abs(err_retained) - abs(err_removed)
    >0 means removing nonverbal annotations improves.
    """
    (ytdrtn, ypdrtn) = dev_retained
    (ytdrmv, ypdrmv) = dev_removed
    (ytt_rtn, ypt_rtn) = test_retained
    (ytt_rmv, ypt_rmv) = test_removed

    if not np.allclose(ytdrtn, ytdrmv):
        print("[WARN] DEV y_true mismatch; paired plot may be invalid if ordering differs.")
    if not np.allclose(ytt_rtn, ytt_rmv):
        print("[WARN] TEST y_true mismatch; paired plot may be invalid if ordering differs.")

    dev_diff = np.abs(ypdrtn - ytdrtn) - np.abs(ypdrmv - ytdrmv)
    test_diff = np.abs(ypt_rtn - ytt_rtn) - np.abs(ypt_rmv - ytt_rmv)

    fig = plt.figure(figsize=(8.0, 5.2))
    ax = fig.add_subplot(111)
    ax.boxplot([dev_diff, test_diff], labels=["DEV", "TEST"], showmeans=True)
    ax.axhline(0, linewidth=1)
    ax.set_ylabel(f"Absolute error difference ({LABEL_RETAINED} − {LABEL_REMOVED})")
    ax.set_title("Per-subject effect (positive = removal improves)")
    fig.tight_layout()
    save_fig(fig, outdir, "fig_paired_absdiff_box")


def plot_scatter_true_pred(dev_retained, dev_removed, test_retained, test_removed, outdir: str) -> None:
    def _one(y_true_rtn, y_pred_rtn, y_true_rmv, y_pred_rmv, title, name):
        lo, hi = diagonal_limits(y_true_rtn, y_pred_rtn, y_true_rmv, y_pred_rmv)
        fig = plt.figure(figsize=(6.2, 6.2))
        ax = fig.add_subplot(111)
        ax.scatter(y_true_rtn, y_pred_rtn, s=25, alpha=0.8, label=LABEL_RETAINED)
        ax.scatter(y_true_rmv, y_pred_rmv, s=25, alpha=0.8, label=LABEL_REMOVED)
        ax.plot([lo, hi], [lo, hi])
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("True PHQ-8")
        ax.set_ylabel("Predicted PHQ-8")
        ax.set_title(title)
        ax.legend(frameon=False)
        fig.tight_layout()
        save_fig(fig, outdir, name)

    (ytdrtn, ypdrtn) = dev_retained
    (ytdrmv, ypdrmv) = dev_removed
    (ytt_rtn, ypt_rtn) = test_retained
    (ytt_rmv, ypt_rmv) = test_removed

    _one(ytdrtn, ypdrtn, ytdrmv, ypdrmv, "DEV: True vs Predicted PHQ-8", "fig_scatter_dev_true_pred")
    _one(ytt_rtn, ypt_rtn, ytt_rmv, ypt_rmv, "TEST: True vs Predicted PHQ-8", "fig_scatter_test_true_pred")


def plot_residuals(dev_retained, dev_removed, test_retained, test_removed, outdir: str) -> None:
    def _one(y_true_rtn, y_pred_rtn, y_true_rmv, y_pred_rmv, title, name):
        res_rtn = y_pred_rtn - y_true_rtn
        res_rmv = y_pred_rmv - y_true_rmv
        fig = plt.figure(figsize=(7.2, 5.2))
        ax = fig.add_subplot(111)
        ax.scatter(y_true_rtn, res_rtn, s=25, alpha=0.8, label=LABEL_RETAINED)
        ax.scatter(y_true_rmv, res_rmv, s=25, alpha=0.8, label=LABEL_REMOVED)
        ax.axhline(0, linewidth=1)
        ax.set_xlabel("True PHQ-8")
        ax.set_ylabel("Residual (Pred − True)")
        ax.set_title(title)
        ax.legend(frameon=False)
        fig.tight_layout()
        save_fig(fig, outdir, name)

    (ytdrtn, ypdrtn) = dev_retained
    (ytdrmv, ypdrmv) = dev_removed
    (ytt_rtn, ypt_rtn) = test_retained
    (ytt_rmv, ypt_rmv) = test_removed

    _one(ytdrtn, ypdrtn, ytdrmv, ypdrmv, "DEV: Residuals vs True PHQ-8", "fig_residuals_dev")
    _one(ytt_rtn, ypt_rtn, ytt_rmv, ypt_rmv, "TEST: Residuals vs True PHQ-8", "fig_residuals_test")


def plot_ecdf_abs_error(dev_retained, dev_removed, test_retained, test_removed, outdir: str) -> None:
    def ecdf(x):
        xs = np.sort(x)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        return xs, ys

    def _one(y_true_rtn, y_pred_rtn, y_true_rmv, y_pred_rmv, title, name):
        ae_rtn = np.abs(y_pred_rtn - y_true_rtn)
        ae_rmv = np.abs(y_pred_rmv - y_true_rmv)
        xrtn, yrtn = ecdf(ae_rtn)
        xrmv, yrmv = ecdf(ae_rmv)

        fig = plt.figure(figsize=(7.2, 5.2))
        ax = fig.add_subplot(111)
        ax.plot(xrtn, yrtn, label=LABEL_RETAINED)
        ax.plot(xrmv, yrmv, label=LABEL_REMOVED)
        ax.set_xlabel("Absolute Error |Pred − True|")
        ax.set_ylabel("ECDF")
        ax.set_title(title)
        ax.legend(frameon=False)
        fig.tight_layout()
        save_fig(fig, outdir, name)

    (ytdrtn, ypdrtn) = dev_retained
    (ytdrmv, ypdrmv) = dev_removed
    (ytt_rtn, ypt_rtn) = test_retained
    (ytt_rmv, ypt_rmv) = test_removed

    _one(ytdrtn, ypdrtn, ytdrmv, ypdrmv, "DEV: ECDF of absolute error", "fig_ecdf_dev_abs_error")
    _one(ytt_rtn, ypt_rtn, ytt_rmv, ypt_rmv, "TEST: ECDF of absolute error", "fig_ecdf_test_abs_error")


def plot_band_mae(dev_retained, dev_removed, test_retained, test_removed, outdir: str) -> None:
    def bandwise_mae(y_true, y_pred, bands):
        out = []
        for lo, hi in bands:
            if hi is None:
                mask = (y_true >= lo)
            else:
                mask = (y_true >= lo) & (y_true <= hi)
            if np.sum(mask) == 0:
                out.append(np.nan)
            else:
                out.append(float(np.mean(np.abs(y_pred[mask] - y_true[mask]))))
        return out

    def _one(y_true_rtn, y_pred_rtn, y_true_rmv, y_pred_rmv, title, name):
        labels = [f"{lo}+" if hi is None else f"{lo}-{hi}" for lo, hi in BANDS]
        v_rtn = bandwise_mae(y_true_rtn, y_pred_rtn, BANDS)
        v_rmv = bandwise_mae(y_true_rmv, y_pred_rmv, BANDS)

        x = np.arange(len(labels))
        width = 0.36
        fig = plt.figure(figsize=(9.0, 5.0))
        ax = fig.add_subplot(111)
        ax.bar(x - width/2, v_rtn, width, label=LABEL_RETAINED)
        ax.bar(x + width/2, v_rmv, width, label=LABEL_REMOVED)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("True PHQ-8 severity band")
        ax.set_ylabel("MAE within band")
        ax.set_title(title)
        ax.legend(frameon=False)
        fig.tight_layout()
        save_fig(fig, outdir, name)

    (ytdrtn, ypdrtn) = dev_retained
    (ytdrmv, ypdrmv) = dev_removed
    (ytt_rtn, ypt_rtn) = test_retained
    (ytt_rmv, ypt_rmv) = test_removed

    _one(ytdrtn, ypdrtn, ytdrmv, ypdrmv, "DEV: MAE by severity band", "fig_band_dev_mae")
    _one(ytt_rtn, ypt_rtn, ytt_rmv, ypt_rmv, "TEST: MAE by severity band", "fig_band_test_mae")


# ---------------------------
# Main
# ---------------------------
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Your files in current folder
    dev_removed_path = os.path.join(base_dir, "")       # removed (text-only)
    dev_retained_path = os.path.join(base_dir, "")    # retained
    test_removed_path = os.path.join(base_dir, "")     # removed (text-only)
    test_retained_path = os.path.join(base_dir, "")  # retained

    for p in [dev_removed_path, dev_retained_path, test_removed_path, test_retained_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"[ERROR] Missing file: {p}")

    # Load
    dev_removed = load_predictions(dev_removed_path)
    dev_retained = load_predictions(dev_retained_path)
    test_removed = load_predictions(test_removed_path)
    test_retained = load_predictions(test_retained_path)

    # Print summary
    def print_summary(name, retained, removed):
        ytrtn, yprtn = retained
        ytrmv, yprmv = removed
        print(f"\n[{name}] N={len(ytrtn)}")
        print(f"  {LABEL_RETAINED}: MAE={mae(ytrtn, yprtn):.4f}, RMSE={rmse(ytrtn, yprtn):.4f}, r={pearson_r(ytrtn, yprtn):.4f}")
        print(f"  {LABEL_REMOVED}:  MAE={mae(ytrmv, yprmv):.4f}, RMSE={rmse(ytrmv, yprmv):.4f}, r={pearson_r(ytrmv, yprmv):.4f}")

    print_summary("DEV", dev_retained, dev_removed)
    print_summary("TEST", test_retained, test_removed)

    # Outdir
    outdir = os.path.join(base_dir, OUTDIR_NAME)
    ensure_outdir(outdir)

    # Generate figures (comment out anything you don't want)
    plot_bar_metrics_ci(dev_retained, dev_removed, test_retained, test_removed, outdir)
    plot_paired_absdiff(dev_retained, dev_removed, test_retained, test_removed, outdir)
    plot_scatter_true_pred(dev_retained, dev_removed, test_retained, test_removed, outdir)
    plot_residuals(dev_retained, dev_removed, test_retained, test_removed, outdir)
    plot_ecdf_abs_error(dev_retained, dev_removed, test_retained, test_removed, outdir)
    plot_band_mae(dev_retained, dev_removed, test_retained, test_removed, outdir)

    print(f"\nDone. Figures saved to: {outdir}")
    print("For paper, typically choose:")
    print("  - fig_bar_metrics_ci.* (main ablation)")
    print("  - fig_paired_absdiff_box.* (paired distribution)")
    print("  - fig_band_*_mae.* (severity-stratified)")

if __name__ == "__main__":
    main()
