import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb


def save_plots(all_preds, all_gt, submission_preds, run_name, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    preds_flat = np.concatenate(all_preds)
    gt_flat = np.concatenate(all_gt)

    sns.set_theme(style="whitegrid")

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(
        gt_flat, bins=100, color="blue", alpha=0.5, label="Ground Truth HR", ax=ax1
    )
    sns.histplot(
        preds_flat, bins=100, color="red", alpha=0.5, label="Predicted HR (CV)", ax=ax1
    )
    ax1.set_title("CV: Ground Truth vs Predicted HR")
    ax1.set_xlabel("Value")
    ax1.legend()
    fig1.savefig(
        os.path.join(plot_dir, "hr_distribution.png"), dpi=150, bbox_inches="tight"
    )
    wandb.log({"hr_distribution": wandb.Image(fig1)})
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.histplot(
        submission_preds,
        bins=100,
        color="red",
        alpha=0.7,
        label="Test Predictions",
        ax=ax2,
    )
    sns.histplot(
        gt_flat, bins=100, color="blue", alpha=0.4, label="Train GT HR (ref)", ax=ax2
    )
    ax2.set_title("Test Predictions vs Train Ground Truth HR")
    ax2.set_xlabel("Value")
    ax2.legend()
    fig2.savefig(
        os.path.join(plot_dir, "test_predictions_distribution.png"),
        dpi=150,
        bbox_inches="tight",
    )
    wandb.log({"test_predictions_distribution": wandb.Image(fig2)})
    plt.close(fig2)

    print(f"Saved plots to {plot_dir}/")


def plot_evaluation_barplots(all_fold_metrics, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    metric_names = ["MAE", "PCC", "JSD", "BC", "EC", "PC"]
    n_folds = len(all_fold_metrics)
    fold_labels = [f"Fold {i + 1}" for i in range(n_folds)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for idx, m in enumerate(metric_names):
        vals = [all_fold_metrics[f][m] for f in range(n_folds)]
        axes[idx].bar(fold_labels, vals, color=sns.color_palette("husl", n_folds))
        axes[idx].set_title(m, fontsize=14)
        for i, v in enumerate(vals):
            axes[idx].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Fold CV Evaluation Metrics", fontsize=16)
    fig.tight_layout()

    path = os.path.join(plot_dir, "evaluation_barplots.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    wandb.log({"evaluation_barplots": wandb.Image(fig)})
    plt.close(fig)
