import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb


def _distribution_subplot(ax, gt_flat, pred_flat, title):
    """Draw GT vs Predicted distribution on a given axis."""
    sns.histplot(gt_flat, bins=100, color="blue", alpha=0.4, label="GT HR", ax=ax, stat="density")
    sns.histplot(pred_flat, bins=100, color="red", alpha=0.4, label="Predicted HR", ax=ax, stat="density")
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.legend()


def _distribution_plot(title, gt_flat, val_flat, test_flat=None):
    """1x2 distribution: (0,0) val vs GT, (0,1) test vs GT."""
    ncols = 2 if test_flat is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    _distribution_subplot(axes[0], gt_flat, val_flat, "Validation: GT vs Predicted")

    if test_flat is not None:
        _distribution_subplot(axes[1], gt_flat, test_flat, "Test: GT (train ref) vs Predicted")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def _heatmap_plot(matrix, title, vmin=0.0, vmax=None):
    """Heatmap of a single connectivity matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, ax=ax, cmap="viridis", square=True, vmin=vmin, vmax=vmax,
                xticklabels=False, yticklabels=False, cbar_kws={"shrink": 0.8})
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _heatmap_comparison(pred_matrix, gt_matrix, title_prefix, sample_idx=0):
    """Side-by-side heatmap: GT vs Predicted vs |Diff|."""
    vmax = max(gt_matrix.max(), pred_matrix.max())
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(gt_matrix, ax=ax1, cmap="viridis", square=True,
                vmin=0, vmax=vmax, xticklabels=False, yticklabels=False)
    ax1.set_title(f"Ground Truth (sample {sample_idx})")

    sns.heatmap(pred_matrix, ax=ax2, cmap="viridis", square=True,
                vmin=0, vmax=vmax, xticklabels=False, yticklabels=False)
    ax2.set_title(f"Predicted (sample {sample_idx})")

    diff = np.abs(pred_matrix - gt_matrix)
    sns.heatmap(diff, ax=ax3, cmap="Reds", square=True,
                vmin=0, xticklabels=False, yticklabels=False)
    ax3.set_title(f"|Predicted - GT| (sample {sample_idx})")

    fig.suptitle(title_prefix, fontsize=14)
    fig.tight_layout()
    return fig


def log_fold_plots(fold_num, fold_pred_matrices, fold_gt_matrices, fold_test_preds=None):
    """Log per-fold distribution + heatmaps to wandb."""
    sns.set_theme(style="whitegrid")

    gt_flat = np.concatenate([g.flatten() for g in fold_gt_matrices])
    val_flat = np.concatenate([p.flatten() for p in fold_pred_matrices])
    test_flat = np.concatenate([p.flatten() for p in fold_test_preds]) if fold_test_preds is not None else None

    # Distribution: 1x2 (val | test)
    fig = _distribution_plot(f"Fold {fold_num}: Distribution", gt_flat, val_flat, test_flat)
    wandb.log({f"fold_{fold_num}/distribution": wandb.Image(fig)})
    plt.close(fig)

    # Val heatmap: GT vs Predicted vs Diff
    fig = _heatmap_comparison(
        fold_pred_matrices[0], fold_gt_matrices[0],
        f"Fold {fold_num}: Validation", sample_idx=0,
    )
    wandb.log({f"fold_{fold_num}/val_heatmap": wandb.Image(fig)})
    plt.close(fig)

    # Test heatmap
    if fold_test_preds is not None:
        fig = _heatmap_plot(fold_test_preds[0], f"Fold {fold_num}: Test Prediction (sample 0)")
        wandb.log({f"fold_{fold_num}/test_heatmap": wandb.Image(fig)})
        plt.close(fig)


def log_average_plots(all_fold_pred_matrices, all_fold_test_preds=None):
    """Log average-across-folds distribution + heatmaps to wandb."""
    sns.set_theme(style="whitegrid")

    all_pred_mats = []
    all_gt_mats = []
    all_pred_flat = []
    all_gt_flat = []
    for fold_preds, fold_gts in all_fold_pred_matrices:
        all_pred_flat.append(np.concatenate([p.flatten() for p in fold_preds]))
        all_gt_flat.append(np.concatenate([g.flatten() for g in fold_gts]))
        all_pred_mats.extend(fold_preds)
        all_gt_mats.extend(fold_gts)

    gt_flat = np.concatenate(all_gt_flat)
    val_flat = np.concatenate(all_pred_flat)

    # Ensemble test predictions
    test_flat = None
    avg_test = None
    if all_fold_test_preds:
        n_test = len(all_fold_test_preds[0])
        avg_test = [
            np.mean([all_fold_test_preds[f][i] for f in range(len(all_fold_test_preds))], axis=0)
            for i in range(n_test)
        ]
        test_flat = np.concatenate([p.flatten() for p in avg_test])

    # Distribution: 1x2 (val | test)
    fig = _distribution_plot("All Folds: Distribution", gt_flat, val_flat, test_flat)
    wandb.log({"avg/distribution": wandb.Image(fig)})
    plt.close(fig)

    # Average val heatmap
    mean_pred = np.mean(all_pred_mats, axis=0)
    mean_gt = np.mean(all_gt_mats, axis=0)
    fig = _heatmap_comparison(mean_pred, mean_gt, "Mean Across All CV Samples")
    wandb.log({"avg/val_heatmap": wandb.Image(fig)})
    plt.close(fig)

    # Average test heatmap
    if avg_test is not None:
        fig = _heatmap_plot(avg_test[0], "Ensemble Test Prediction (sample 0)")
        wandb.log({"avg/test_heatmap": wandb.Image(fig)})
        plt.close(fig)


def log_mae_mse_barplot(fold_mse, fold_mae):
    """2x2 bar plot: Fold 1, Fold 2, Fold 3, Average+Std for MAE and MSE."""
    n_folds = len(fold_mse)
    colors = sns.color_palette("husl", 2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Per-fold subplots
    for i in range(n_folds):
        row, col = divmod(i, 2)
        ax = axes[row][col]
        bars = ax.bar(["MAE", "MSE"], [fold_mae[i], fold_mse[i]], color=colors)
        ax.set_title(f"Fold {i + 1}", fontsize=13)
        ax.set_ylim(0, max(max(fold_mae), max(fold_mse)) * 1.3)
        for bar, val in zip(bars, [fold_mae[i], fold_mse[i]]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.5f}", ha="center", va="bottom", fontsize=10)

    # Average + Std subplot (bottom-right)
    ax = axes[1][1]
    avg_mae, std_mae = np.mean(fold_mae), np.std(fold_mae)
    avg_mse, std_mse = np.mean(fold_mse), np.std(fold_mse)
    bars = ax.bar(["MAE", "MSE"], [avg_mae, avg_mse], yerr=[std_mae, std_mse],
                  color=colors, capsize=8)
    ax.set_title("Average (+/- Std)", fontsize=13)
    ax.set_ylim(0, max(max(fold_mae), max(fold_mse)) * 1.3)
    for bar, val, std in zip(bars, [avg_mae, avg_mse], [std_mae, std_mse]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std,
                f"{val:.5f}\n\u00b1{std:.5f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("MAE / MSE per Fold", fontsize=15)
    fig.tight_layout()

    wandb.log({"cv/mae_mse_barplot": wandb.Image(fig)})
    plt.close(fig)


def plot_evaluation_barplots(all_fold_metrics):
    """Log evaluation bar plots to wandb."""
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

    wandb.log({"evaluation_barplots": wandb.Image(fig)})
    plt.close(fig)
