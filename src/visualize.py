"""
visualize.py
------------
All plotting utilities for the self-pruning project.

Plots generated
---------------
1. gate_histogram          – histogram of gate values across all layers
2. sparsity_vs_epoch       – sparsity % over training epochs
3. accuracy_vs_epoch        – train & val accuracy over training epochs
4. loss_vs_epoch            – train & val loss over training epochs (extra insight)
5. layer_sparsity_bar       – per-layer sparsity at end of training
6. weight_heatmap           – heatmap of a single FC layer's weight × gate product
7. lambda_comparison        – bar/line comparison across multiple λ runs
8. gate_distribution_layers – per-layer gate distribution as KDE / histogram

All functions accept a *save_dir* keyword argument. When provided the figure
is saved as a PNG to that directory.  When *show* is True the figure is also
displayed interactively.
"""

from __future__ import annotations
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on all platforms)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import torch

from src.layers import PrunableLinear, PrunableConv2d


# ---------------------------------------------------------------------------
# Style & colour helpers
# ---------------------------------------------------------------------------

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]

def _setup_style():
    plt.rcParams.update({
        "figure.dpi":       150,
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "font.family":      "DejaVu Sans",
        "font.size":        11,
        "axes.titlesize":   13,
        "axes.labelsize":   11,
        "legend.fontsize":  10,
    })

def _save(fig, save_dir: str, filename: str, show: bool):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        print(f"[Plot] Saved → {path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Gate histogram
# ---------------------------------------------------------------------------

def plot_gate_histogram(
    model,
    title: str = "Gate Value Distribution",
    save_dir: str = None,
    show: bool = False,
):
    _setup_style()
    all_gates = []
    for m in model.modules():
        if isinstance(m, (PrunableLinear, PrunableConv2d)):
            g = m.get_gates().cpu().numpy().ravel()
            all_gates.append(g)
    if not all_gates:
        print("[visualize] No prunable layers found.")
        return

    gates = np.concatenate(all_gates)

    # Guard: if all gates are identical (e.g. early training), use fewer bins
    gate_range = gates.max() - gates.min()
    n_bins = 60 if gate_range > 1e-6 else 1

    fig, ax = plt.subplots(figsize=(8, 4))
    counts, bins, patches = ax.hist(gates, bins=n_bins, color=PALETTE[0], edgecolor="white", linewidth=0.4)
    ax.axvline(x=0.5, color="#C44E52", linestyle="--", linewidth=1.5, label="Threshold 0.5")
    ax.axvline(x=0.01, color="#55A868", linestyle="--", linewidth=1.5, label="ε = 0.01")
    ax.set_xlabel("Gate Value  sigmoid(score / T)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    # annotate sparsity
    sparsity = (gates < 0.01).mean() * 100
    ax.text(0.02, 0.92, f"Sparsity (gates < 0.01): {sparsity:.1f}%",
            transform=ax.transAxes, fontsize=10, color="#C44E52")

    _save(fig, save_dir, "gate_histogram.png", show)


# ---------------------------------------------------------------------------
# 2. Sparsity vs Epoch
# ---------------------------------------------------------------------------

def plot_sparsity_vs_epoch(
    histories: dict[str, dict],
    save_dir: str = None,
    show: bool = False,
):
    """histories = {label: history_dict}"""
    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, hist) in enumerate(histories.items()):
        epochs    = range(1, len(hist["sparsity"]) + 1)
        sparsity  = [s * 100 for s in hist["sparsity"]]
        ax.plot(epochs, sparsity, label=label, color=PALETTE[i % len(PALETTE)], linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Global Sparsity (%)")
    ax.set_title("Sparsity Evolution over Training")
    ax.legend(title="Config")
    _save(fig, save_dir, "sparsity_vs_epoch.png", show)


# ---------------------------------------------------------------------------
# 3. Accuracy vs Epoch
# ---------------------------------------------------------------------------

def plot_accuracy_vs_epoch(
    histories: dict[str, dict],
    save_dir: str = None,
    show: bool = False,
):
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for i, (label, hist) in enumerate(histories.items()):
        epochs   = range(1, len(hist["train_acc"]) + 1)
        col      = PALETTE[i % len(PALETTE)]
        axes[0].plot(epochs, [a * 100 for a in hist["train_acc"]],
                     label=label, color=col, linewidth=2)
        axes[1].plot(epochs, [a * 100 for a in hist["val_acc"]],
                     label=label, color=col, linewidth=2)

    for ax, title in zip(axes, ["Train Accuracy (%)", "Validation Accuracy (%)"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title)
        ax.legend(title="Config")

    fig.suptitle("Accuracy vs Epoch", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_dir, "accuracy_vs_epoch.png", show)


# ---------------------------------------------------------------------------
# 4. Loss vs Epoch
# ---------------------------------------------------------------------------

def plot_loss_vs_epoch(
    histories: dict[str, dict],
    save_dir: str = None,
    show: bool = False,
):
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for i, (label, hist) in enumerate(histories.items()):
        epochs = range(1, len(hist["train_loss"]) + 1)
        col    = PALETTE[i % len(PALETTE)]
        axes[0].plot(epochs, hist["train_loss"], label=label, color=col, linewidth=2)
        axes[1].plot(epochs, hist["val_loss"],   label=label, color=col, linewidth=2)

    for ax, title in zip(axes, ["Train Loss", "Validation Loss"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend(title="Config")

    fig.suptitle("Loss vs Epoch", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_dir, "loss_vs_epoch.png", show)


# ---------------------------------------------------------------------------
# 5. Layer-wise sparsity bar chart
# ---------------------------------------------------------------------------

def plot_layer_sparsity(
    sparsity_dict: dict[str, float],
    title: str = "Layer-wise Sparsity",
    save_dir: str = None,
    show: bool = False,
):
    _setup_style()
    names   = list(sparsity_dict.keys())
    values  = [v * 100 for v in sparsity_dict.values()]
    colors  = [PALETTE[i % len(PALETTE)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.4), 5))
    bars = ax.bar(names, values, color=colors, edgecolor="white", width=0.6)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_ylabel("Sparsity (%)")
    ax.set_title(title)
    ax.set_ylim(0, 110)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, save_dir, "layer_sparsity_bar.png", show)


# ---------------------------------------------------------------------------
# 6. Weight heatmap (first PrunableLinear layer)
# ---------------------------------------------------------------------------

def plot_weight_heatmap(
    model,
    layer_name: str = None,
    save_dir: str = None,
    show: bool = False,
    max_rows: int = 64,
    max_cols: int = 64,
):
    _setup_style()
    target = None
    t_name = ""
    for name, m in model.named_modules():
        if isinstance(m, PrunableLinear):
            if layer_name is None or name == layer_name:
                target = m
                t_name = name
                break

    if target is None:
        print("[visualize] No matching PrunableLinear layer found for heatmap.")
        return

    with torch.no_grad():
        gates   = target.get_gates()                       # (out, in)
        w_gated = (target.weight * gates).cpu().numpy()   # effective weight

    # Subsample for readability
    w_plot = w_gated[:max_rows, :max_cols]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title_suffix in zip(
        axes,
        [target.weight.detach().cpu().numpy()[:max_rows, :max_cols], w_plot],
        ["Raw Weights", "Pruned Weights (weight × gate)"],
    ):
        vmax = np.abs(data).max() or 1.0
        im = ax.imshow(data, cmap="RdBu_r", aspect="auto",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"[{t_name}]  {title_suffix}")
        ax.set_xlabel("Input neuron index")
        ax.set_ylabel("Output neuron index")

    fig.suptitle(f"Weight Heatmap – Layer: {t_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_dir, "weight_heatmap.png", show)


# ---------------------------------------------------------------------------
# 7. Lambda comparison summary plot
# ---------------------------------------------------------------------------

def plot_lambda_comparison(
    results: list[dict],
    save_dir: str = None,
    show: bool = False,
):
    """
    results = [
        {"lambda": 0.0001, "accuracy": 0.88, "sparsity": 12.3,
         "params_before": 5e6, "params_after": 4.5e6, "infer_ms": 2.1},
        ...
    ]
    """
    _setup_style()
    lambdas   = [r["lambda"]   for r in results]
    accs      = [r["test_acc"] for r in results]
    sparsities = [r["sparsity_pct"] for r in results]
    labels    = [f"λ={lam}" for lam in lambdas]

    x = np.arange(len(lambdas))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Accuracy
    bars1 = axes[0].bar(x, accs, width, color=PALETTE[:len(lambdas)], edgecolor="white")
    axes[0].bar_label(bars1, fmt="%.1f%%", padding=3)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].set_title("Accuracy vs Lambda")
    axes[0].set_ylim(0, 100)

    # Sparsity
    bars2 = axes[1].bar(x, sparsities, width, color=PALETTE[1:len(lambdas)+1], edgecolor="white")
    axes[1].bar_label(bars2, fmt="%.1f%%", padding=3)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Sparsity (gates < 0.01)  %")
    axes[1].set_title("Sparsity vs Lambda")
    axes[1].set_ylim(0, 100)

    fig.suptitle("Effect of Sparsity Regularisation (λ)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_dir, "lambda_comparison.png", show)


# ---------------------------------------------------------------------------
# 8. Per-layer gate distribution (multi-panel)
# ---------------------------------------------------------------------------

def plot_gate_distributions_per_layer(
    model,
    save_dir: str = None,
    show: bool = False,
):
    _setup_style()
    layers = [
        (name, m) for name, m in model.named_modules()
        if isinstance(m, (PrunableLinear, PrunableConv2d))
    ]
    if not layers:
        return

    ncols = min(3, len(layers))
    nrows = math.ceil(len(layers) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5),
                             constrained_layout=True)
    if len(layers) == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]

    for idx, (name, m) in enumerate(layers):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        g  = m.get_gates().cpu().numpy().ravel()
        n_bins_layer = 40 if (g.max() - g.min()) > 1e-6 else 1
        ax.hist(g, bins=n_bins_layer, color=PALETTE[idx % len(PALETTE)], edgecolor="white", linewidth=0.3)
        ax.axvline(0.5, color="#C44E52", linestyle="--", linewidth=1.2)
        sp = (g < 0.01).mean() * 100
        ax.set_title(f"{name}\nSparsity={sp:.1f}%", fontsize=9)
        ax.set_xlabel("Gate"); ax.set_ylabel("Count")

    # hide unused subplots
    for idx in range(len(layers), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Gate Value Distribution per Layer", fontsize=13, fontweight="bold")
    _save(fig, save_dir, "gate_distributions_per_layer.png", show)
