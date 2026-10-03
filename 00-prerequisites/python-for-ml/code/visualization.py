"""
Python for ML - Visualization Utilities
Clean, publication-ready plotting routines using Matplotlib and Seaborn.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def set_academic_style() -> None:
    """Configures clean typography, grid lines, and aesthetic defaults."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 2.0,
        "figure.autolayout": True,
    })


def plot_learning_curves(
    train_losses: list[float],
    val_losses: list[float],
    title: str = "Training and Validation Loss",
    save_path: str | None = None
) -> plt.Figure:
    """
    Renders training vs validation loss curves over epochs.
    """
    set_academic_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Training Loss", color="#1f77b4")
    ax.plot(epochs, val_losses, label="Validation Loss", color="#d62728", linestyle="--")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save_path: str | None = None
) -> plt.Figure:
    """
    Plots a confusion matrix heatmap with normalized percentages and counts.
    """
    set_academic_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    if normalize:
        cm_display = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax
    )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
