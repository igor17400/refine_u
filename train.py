"""
Unified training script with Hydra configuration.

Usage (from inside refine_u/):
    python -m train                                            # default (traditional GCN)
    python -m train experiment=refine_u                        # best UNet config
    python -m train experiment=gsrn_baseline                   # GSRN baseline
    python -m train model.gcn_type=unet training.epochs=100    # inline overrides
"""

import copy
import os

import matplotlib

matplotlib.use("Agg")

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import KFold

from refine_u.data import (
    GraphCentralityDataset,
    augment_adjacency,
    load_data,
)
from refine_u.evaluate import (
    compute_full_evaluation,
    compute_loss,
    mae_criterion,
    mse_criterion,
    save_submission_csv,
)
from refine_u.models import GraphCentMapperModel, GSRNet
from refine_u.models.gsrn import pad_HR_adj, unpad

# ---------------------------------------------------------------------------
# GraphCent training helpers
# ---------------------------------------------------------------------------


def train_one_epoch_gc(model, optimizer, adj_data, feat_data, labels, device, cfg):
    epoch_mse, epoch_mae = [], []

    for adj, feat, hr in zip(adj_data, feat_data, labels):
        model.train()
        optimizer.zero_grad()

        adj = adj.unsqueeze(0).to(device)
        feat = feat.unsqueeze(0).to(device)
        hr = hr.unsqueeze(0).to(device)

        if cfg.training.aug_noise > 0 or cfg.training.aug_edge_drop > 0:
            adj = augment_adjacency(
                adj,
                noise_std=cfg.training.aug_noise,
                edge_drop=cfg.training.aug_edge_drop,
            )

        pred, x_in, x_out = model(adj, feat)

        pred_loss = compute_loss(pred, hr, cfg.training.loss)
        unet_loss = mse_criterion(x_in, x_out)
        loss = pred_loss + (
            cfg.model.unet_loss_weight * unet_loss if model.gcn_type == "unet" else 0.0
        )

        loss.backward()
        optimizer.step()

        epoch_mse.append(mse_criterion(pred, hr).item())
        epoch_mae.append(mae_criterion(pred, hr).item())

    return np.mean(epoch_mse), np.mean(epoch_mae)


def evaluate_gc(model, adj_data, feat_data, labels, device):
    mse_error, mae_error = [], []

    model.eval()
    with torch.no_grad():
        for adj, feat, hr in zip(adj_data, feat_data, labels):
            if not torch.any(adj) or not torch.any(hr):
                continue

            adj = adj.unsqueeze(0).to(device)
            feat = feat.unsqueeze(0).to(device)
            hr = hr.unsqueeze(0).to(device)

            preds, _, _ = model(adj, feat)

            mse_error.append(mse_criterion(preds, hr).item())
            mae_error.append(mae_criterion(preds, hr).item())

    return np.mean(mse_error), np.mean(mae_error)


def predict_gc(model, adj_data, feat_data, device):
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(adj_data.shape[0]):
            adj = adj_data[i].unsqueeze(0).to(device)
            feat = feat_data[i].unsqueeze(0).to(device)
            pred, _, _ = model(adj, feat)
            preds.append(pred[0].cpu().numpy())
    return preds


# ---------------------------------------------------------------------------
# GSRN training helpers
# ---------------------------------------------------------------------------


def train_one_epoch_gsrn(model, optimizer, lr_data, hr_data, device, cfg):
    epoch_mse, epoch_mae = [], []
    padding = cfg.model.padding

    for lr, hr in zip(lr_data, hr_data):
        model.train()
        optimizer.zero_grad()

        lr = lr.to(device)
        hr_np = hr.cpu().numpy()

        model_outputs, net_outs, start_gcn_outs, layer_outs = model(lr)
        model_outputs_unpadded = unpad(model_outputs, padding)

        padded_hr = pad_HR_adj(hr_np, padding).to(device)
        eig_val_hr, U_hr = torch.linalg.eigh(padded_hr, UPLO="U")

        hr = hr.to(device)
        loss = (
            cfg.model.lmbda * mse_criterion(net_outs, start_gcn_outs)
            + mse_criterion(model.layer.weights, U_hr)
            + mse_criterion(model_outputs_unpadded, hr)
        )

        loss.backward()
        optimizer.step()

        epoch_mse.append(mse_criterion(model_outputs_unpadded, hr).item())
        epoch_mae.append(mae_criterion(model_outputs_unpadded, hr).item())

    return np.mean(epoch_mse), np.mean(epoch_mae)


def evaluate_gsrn(model, lr_data, hr_data, device, padding):
    mse_error, mae_error = [], []

    model.eval()
    with torch.no_grad():
        for lr, hr in zip(lr_data, hr_data):
            if not torch.any(lr) or not torch.any(hr):
                continue

            lr = lr.to(device)
            hr = hr.to(device)

            preds, _, _, _ = model(lr)
            preds = unpad(preds, padding)

            mse_error.append(mse_criterion(preds, hr).item())
            mae_error.append(mae_criterion(preds, hr).item())

    return np.mean(mse_error), np.mean(mae_error)


def predict_gsrn(model, lr_data, device, padding):
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(lr_data.shape[0]):
            lr = lr_data[i].to(device)
            pred, _, _, _ = model(lr)
            pred = unpad(pred, padding)
            preds.append(pred.cpu().numpy())
    return preds


# ---------------------------------------------------------------------------
# Model / feature builders
# ---------------------------------------------------------------------------


def build_gc_model(cfg, device):
    gcn_in = {"eigen": 16, "betweenness": 1, "both": 17, "ones": cfg.data.lr_dim}[
        cfg.model.feat_type
    ]
    pool_ratios = (
        list(cfg.model.pool_ratios) if cfg.model.pool_ratios is not None else None
    )
    return GraphCentMapperModel(
        input_nodes=cfg.data.lr_dim,
        output_nodes=cfg.data.hr_dim,
        gcn_in=gcn_in,
        gcn_hidden=cfg.model.gcn_hidden,
        gcn_out=cfg.model.gcn_out,
        mlp_hidden=cfg.model.mlp_hidden,
        dropout=cfg.model.dropout,
        gcn_type=cfg.model.gcn_type,
        pool_ratios=pool_ratios,
        gated_skip=cfg.model.gated_skip,
        low_rank_k=cfg.model.low_rank_k,
    ).to(device)


def build_gsrn_model(cfg, device):
    ks = list(cfg.model.pool_ratios)
    return GSRNet(
        ks=ks,
        lr_dim=cfg.data.lr_dim,
        hr_dim=cfg.data.hr_dim,
        hidden_dim=cfg.model.hidden_dim,
        padding=cfg.model.padding,
    ).to(device)


def get_features(cfg, lr_matrices, hr_matrices, lr_matrices_test):
    cache = cfg.data.cache_dir

    print("Preparing features for training data...")
    train_ds = GraphCentralityDataset(
        lr_matrices, hr_matrices, save_path=os.path.join(cache, "cache_train")
    )
    print("Preparing features for test data...")
    test_ds = GraphCentralityDataset(
        lr_matrices_test, targets=None, save_path=os.path.join(cache, "cache_test")
    )

    feat_type = cfg.model.feat_type
    if feat_type == "ones":
        n_nodes = cfg.data.lr_dim
        train_f = torch.ones(lr_matrices.shape[0], n_nodes, n_nodes)
        test_f = torch.ones(lr_matrices_test.shape[0], n_nodes, n_nodes)
        return train_f, test_f
    elif feat_type == "eigen":
        return train_ds.e_features, test_ds.e_features
    elif feat_type == "betweenness":
        return train_ds.b_features, test_ds.b_features
    else:  # both
        train_f = torch.cat([train_ds.e_features, train_ds.b_features], dim=-1)
        test_f = torch.cat([test_ds.e_features, test_ds.b_features], dim=-1)
        return train_f, test_f


# ---------------------------------------------------------------------------
# Run name builder
# ---------------------------------------------------------------------------


def build_run_name(cfg):
    parts = [cfg.model.name]

    if cfg.model.name == "graph_cent":
        parts.append(cfg.model.gcn_type)
        parts.append(cfg.model.feat_type)
        if cfg.model.gcn_type == "unet" and cfg.model.pool_ratios:
            pool_str = "_".join(str(r) for r in cfg.model.pool_ratios)
            parts.append(f"p{pool_str}")
        if cfg.model.gated_skip:
            parts.append("gated")
        if cfg.model.low_rank_k > 0:
            parts.append(f"lr{cfg.model.low_rank_k}")
    elif cfg.model.name == "gsrn":
        parts.append(f"lmbda{cfg.model.lmbda}")

    parts.append(f"{cfg.training.epochs}ep")
    parts.append(f"{cfg.training.splits}f")

    seeds = list(cfg.training.seeds)
    if len(seeds) > 1:
        parts.append(f"{len(seeds)}s")

    parts.append(cfg.training.loss)
    parts.append("ens" if cfg.training.ensemble else "full")

    return "-".join(parts)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Main: GraphCent pipeline
# ---------------------------------------------------------------------------


def run_graph_cent(cfg, lr_matrices, hr_matrices, lr_matrices_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_feats, test_feats = get_features(
        cfg, lr_matrices, hr_matrices, lr_matrices_test
    )

    seeds = list(cfg.training.seeds)
    total_models = len(seeds) * cfg.training.splits

    print(f"\n{'=' * 60}")
    print(
        f"Phase 1: {cfg.training.splits}-fold CV x {len(seeds)} seeds = {total_models} models"
    )
    print(f"{'=' * 60}")

    fold_mse, fold_mae = [], []
    all_preds, all_gt = [], []
    fold_test_preds = []
    all_fold_pred_matrices = []  # (fold_preds, fold_gts) for deferred evaluation
    global_step = 0
    model_idx = 0

    for seed in seeds:
        print(f"\nSeed: {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        cv = KFold(n_splits=cfg.training.splits, shuffle=True, random_state=seed)

        for fold_idx, (train_index, test_index) in enumerate(cv.split(lr_matrices)):
            model_idx += 1
            print(
                f"\n--- Seed {seed}, Fold {fold_idx + 1}/{cfg.training.splits} (model {model_idx}/{total_models}) ---"
            )

            model = build_gc_model(cfg, device)

            if model_idx == 1:
                print(model)
                n_params = sum(p.numel() for p in model.parameters())
                print(f"Total parameters: {n_params / 1e6:.4f}M")
                wandb.config.update({"n_params": n_params})

            optimizer = optim.Adam(
                model.parameters(),
                lr=cfg.training.lr,
                weight_decay=cfg.training.weight_decay,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.training.epochs, eta_min=cfg.training.eta_min
            )

            x_train = lr_matrices[train_index]
            f_train = train_feats[train_index]
            y_train = hr_matrices[train_index]
            x_val = lr_matrices[test_index]
            f_val = train_feats[test_index]
            y_val = hr_matrices[test_index]

            best_val_mae = float("inf")
            best_state = None
            wait = 0

            for epoch in range(cfg.training.epochs):
                train_mse, train_mae = train_one_epoch_gc(
                    model, optimizer, x_train, f_train, y_train, device, cfg
                )
                val_mse, val_mae = evaluate_gc(model, x_val, f_val, y_val, device)
                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()

                if cfg.training.best_ckpt and val_mae < best_val_mae - 1e-5:
                    best_val_mae = val_mae
                    best_state = copy.deepcopy(model.state_dict())
                    wait = 0
                elif cfg.training.patience > 0:
                    wait += 1
                    if wait >= cfg.training.patience:
                        print(f"  Early stopping at epoch {epoch + 1}")
                        break

                global_step += 1
                wandb.log(
                    {
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "seed": seed,
                        "fold": fold_idx + 1,
                        "train/mse": train_mse,
                        "train/mae": train_mae,
                        "val/mse": val_mse,
                        "val/mae": val_mae,
                        "lr": current_lr,
                    },
                    step=global_step,
                )

                if (epoch + 1) % 10 == 0 or epoch == cfg.training.epochs - 1:
                    print(
                        f"Epoch {epoch + 1:3d} | Train MSE: {train_mse:.6f} MAE: {train_mae:.6f} "
                        f"| Val MSE: {val_mse:.6f} MAE: {val_mae:.6f}"
                    )

            if cfg.training.best_ckpt and best_state is not None:
                model.load_state_dict(best_state)
                print(f"  Restored best checkpoint (val MAE: {best_val_mae:.6f})")

            val_mse, val_mae = evaluate_gc(model, x_val, f_val, y_val, device)
            fold_mse.append(val_mse)
            fold_mae.append(val_mae)

            fold_key = (
                f"s{seed}_f{fold_idx + 1}" if len(seeds) > 1 else f"fold_{fold_idx + 1}"
            )
            wandb.log({f"{fold_key}/val_mse": val_mse, f"{fold_key}/val_mae": val_mae})

            # Collect CV predictions (defer full evaluation to after all folds)
            fold_pred_matrices, fold_gt_matrices = [], []
            model.eval()
            with torch.no_grad():
                for adj_s, feat_s, hr_s in zip(x_val, f_val, y_val):
                    if not torch.any(adj_s) or not torch.any(hr_s):
                        continue
                    adj_s = adj_s.unsqueeze(0).to(device)
                    feat_s = feat_s.unsqueeze(0).to(device)
                    hr_s = hr_s.unsqueeze(0).to(device)
                    pred_s, _, _ = model(adj_s, feat_s)
                    fold_pred_matrices.append(
                        np.clip(pred_s[0].cpu().numpy(), 0.0, 1.0)
                    )
                    fold_gt_matrices.append(hr_s[0].cpu().numpy())
                    all_preds.append(pred_s.cpu().flatten().numpy())
                    all_gt.append(hr_s.cpu().flatten().numpy())

            all_fold_pred_matrices.append(
                (np.array(fold_pred_matrices), np.array(fold_gt_matrices))
            )

            print(
                f"Seed {seed}, Fold {fold_idx + 1} — MSE: {val_mse:.6f} MAE: {val_mae:.6f}"
            )

            if cfg.training.ensemble:
                fold_preds = predict_gc(model, lr_matrices_test, test_feats, device)
                fold_test_preds.append(fold_preds)

    # CV summary
    _print_cv_summary(seeds, cfg.training.splits, fold_mse, fold_mae)
    avg_mae = np.mean(fold_mae)
    wandb.log(
        {
            "cv/avg_mse": np.mean(fold_mse),
            "cv/avg_mae": avg_mae,
            "cv/std_mse": np.std(fold_mse),
            "cv/std_mae": np.std(fold_mae),
        }
    )

    run_name = build_run_name(cfg)

    # Save per-fold prediction CSVs and compute deferred evaluation
    print(f"\n{'=' * 60}")
    print("Post-training: saving fold CSVs and computing evaluation metrics")
    print(f"{'=' * 60}")

    fold_output_dir = os.path.join("outputs", run_name, "folds")
    os.makedirs(fold_output_dir, exist_ok=True)

    all_fold_metrics = []
    fold_artifact = wandb.Artifact(
        name=f"fold-predictions-{run_name}"[:128],
        type="fold_predictions",
        description=f"Per-fold CV prediction CSVs — {run_name}",
    )

    for fold_i, (fold_preds, fold_gts) in enumerate(all_fold_pred_matrices):
        fold_num = fold_i + 1

        # Save fold prediction CSV in spec format
        fold_csv_path = os.path.join(
            fold_output_dir, f"predictions_fold_{fold_num}.csv"
        )
        save_submission_csv(fold_preds, fold_csv_path)
        fold_artifact.add_file(fold_csv_path, name=f"predictions_fold_{fold_num}.csv")

        # Compute full evaluation (deferred — runs once after training)
        fold_metrics = compute_full_evaluation(fold_preds, fold_gts)
        all_fold_metrics.append(fold_metrics)
        print(
            f"  Fold {fold_num}: MAE={fold_metrics['MAE']:.5f} PCC={fold_metrics['PCC']:.5f} "
            f"JSD={fold_metrics['JSD']:.5f} BC={fold_metrics['BC']:.5f} "
            f"EC={fold_metrics['EC']:.5f} PC={fold_metrics['PC']:.5f}"
        )

    wandb.log_artifact(fold_artifact)

    if all_fold_metrics:
        _print_eval_summary(all_fold_metrics)
        avg_metrics = {
            k: np.mean([m[k] for m in all_fold_metrics]) for k in all_fold_metrics[0]
        }
        wandb.log({f"cv/avg_{k.lower()}": v for k, v in avg_metrics.items()})

    # Phase 2: Generate submission
    if cfg.training.ensemble:
        pred_matrices = _ensemble_predictions(
            fold_test_preds, lr_matrices_test.shape[0]
        )
    else:
        pred_matrices = _full_retrain_gc(
            cfg,
            lr_matrices,
            hr_matrices,
            lr_matrices_test,
            train_feats,
            test_feats,
            device,
            global_step,
        )

    # Rescale if requested
    if cfg.training.rescale:
        pred_matrices = _rescale_predictions(pred_matrices, hr_matrices)

    submission_path = os.path.join("outputs", run_name, f"{run_name}_predictions.csv")
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df = save_submission_csv(pred_matrices, submission_path)

    # Plots
    plot_dir = os.path.join("outputs", run_name, "plots")
    if all_fold_metrics:
        plot_evaluation_barplots(all_fold_metrics, plot_dir)
    save_plots(all_preds, all_gt, submission_df["Predicted"].values, run_name, plot_dir)

    # Wandb artifact
    artifact = wandb.Artifact(
        name=f"submission-{run_name}"[:128],
        type="submission",
        description=f"CV avg MAE: {avg_mae:.6f}",
    )
    artifact.add_file(submission_path)
    wandb.log_artifact(artifact)


# ---------------------------------------------------------------------------
# Main: GSRN pipeline
# ---------------------------------------------------------------------------


def run_gsrn(cfg, lr_matrices, hr_matrices, lr_matrices_test):
    # GSRN uses plain Python lists (not nn.ModuleList), so .to(device) doesn't propagate
    device = torch.device("cpu")
    print(f"Using device: {device} (forced CPU for GSRN)")

    seeds = list(cfg.training.seeds)
    padding = cfg.model.padding

    fold_mse, fold_mae = [], []
    all_preds, all_gt = [], []
    fold_test_preds = []
    all_fold_pred_matrices = []
    global_step = 0

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # GSRN shares a single model across folds (matches original)
        model = build_gsrn_model(cfg, device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )

        if seed == seeds[0]:
            print(model)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"Total parameters: {n_params / 1e6:.4f}M")
            wandb.config.update({"n_params": n_params})

        cv = KFold(n_splits=cfg.training.splits, shuffle=True, random_state=seed)

        for fold_idx, (train_index, test_index) in enumerate(cv.split(lr_matrices)):
            print(f"\n--- Seed {seed}, Fold {fold_idx + 1}/{cfg.training.splits} ---")

            x_train = lr_matrices[train_index]
            y_train = hr_matrices[train_index]
            x_val = lr_matrices[test_index]
            y_val = hr_matrices[test_index]

            for epoch in range(cfg.training.epochs):
                train_mse, train_mae = train_one_epoch_gsrn(
                    model, optimizer, x_train, y_train, device, cfg
                )
                val_mse, val_mae = evaluate_gsrn(model, x_val, y_val, device, padding)

                global_step += 1
                wandb.log(
                    {
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "fold": fold_idx + 1,
                        "train/mse": train_mse,
                        "train/mae": train_mae,
                        "val/mse": val_mse,
                        "val/mae": val_mae,
                    },
                    step=global_step,
                )

                if (epoch + 1) % 10 == 0 or epoch == cfg.training.epochs - 1:
                    print(
                        f"Epoch {epoch + 1:3d} | Train MSE: {train_mse:.6f} MAE: {train_mae:.6f} "
                        f"| Val MSE: {val_mse:.6f} MAE: {val_mae:.6f}"
                    )

            val_mse, val_mae = evaluate_gsrn(model, x_val, y_val, device, padding)
            fold_mse.append(val_mse)
            fold_mae.append(val_mae)

            # CV predictions (defer evaluation to after all folds)
            fold_pred_matrices, fold_gt_matrices = [], []
            model.eval()
            with torch.no_grad():
                for lr_s, hr_s in zip(x_val, y_val):
                    if not torch.any(lr_s) or not torch.any(hr_s):
                        continue
                    lr_s = lr_s.to(device)
                    hr_s = hr_s.to(device)
                    pred_s, _, _, _ = model(lr_s)
                    pred_s = unpad(pred_s, padding)
                    pred_np = np.clip(pred_s.cpu().numpy(), 0.0, 1.0)
                    fold_pred_matrices.append(pred_np)
                    fold_gt_matrices.append(hr_s.cpu().numpy())
                    all_preds.append(pred_s.cpu().flatten().numpy())
                    all_gt.append(hr_s.cpu().flatten().numpy())

            all_fold_pred_matrices.append(
                (np.array(fold_pred_matrices), np.array(fold_gt_matrices))
            )

            print(f"Fold {fold_idx + 1} — MSE: {val_mse:.6f} MAE: {val_mae:.6f}")

            if cfg.training.ensemble:
                preds = predict_gsrn(model, lr_matrices_test, device, padding)
                fold_test_preds.append(preds)

    _print_cv_summary(seeds, cfg.training.splits, fold_mse, fold_mae)
    avg_mae = np.mean(fold_mae)
    wandb.log(
        {
            "cv/avg_mse": np.mean(fold_mse),
            "cv/avg_mae": avg_mae,
            "cv/std_mse": np.std(fold_mse),
            "cv/std_mae": np.std(fold_mae),
        }
    )

    run_name = build_run_name(cfg)

    # Save per-fold prediction CSVs and compute deferred evaluation
    print(f"\n{'=' * 60}")
    print("Post-training: saving fold CSVs and computing evaluation metrics")
    print(f"{'=' * 60}")

    fold_output_dir = os.path.join("outputs", run_name, "folds")
    os.makedirs(fold_output_dir, exist_ok=True)

    all_fold_metrics = []
    fold_artifact = wandb.Artifact(
        name=f"fold-predictions-{run_name}"[:128],
        type="fold_predictions",
        description=f"Per-fold CV prediction CSVs — {run_name}",
    )

    for fold_i, (fold_preds, fold_gts) in enumerate(all_fold_pred_matrices):
        fold_num = fold_i + 1
        fold_csv_path = os.path.join(
            fold_output_dir, f"predictions_fold_{fold_num}.csv"
        )
        save_submission_csv(fold_preds, fold_csv_path)
        fold_artifact.add_file(fold_csv_path, name=f"predictions_fold_{fold_num}.csv")

        fold_metrics = compute_full_evaluation(fold_preds, fold_gts)
        all_fold_metrics.append(fold_metrics)
        print(
            f"  Fold {fold_num}: MAE={fold_metrics['MAE']:.5f} PCC={fold_metrics['PCC']:.5f} "
            f"JSD={fold_metrics['JSD']:.5f} BC={fold_metrics['BC']:.5f} "
            f"EC={fold_metrics['EC']:.5f} PC={fold_metrics['PC']:.5f}"
        )

    wandb.log_artifact(fold_artifact)

    if all_fold_metrics:
        _print_eval_summary(all_fold_metrics)
        avg_metrics = {
            k: np.mean([m[k] for m in all_fold_metrics]) for k in all_fold_metrics[0]
        }
        wandb.log({f"cv/avg_{k.lower()}": v for k, v in avg_metrics.items()})

    if cfg.training.ensemble:
        pred_matrices = _ensemble_predictions(
            fold_test_preds, lr_matrices_test.shape[0]
        )
    else:
        pred_matrices = predict_gsrn(model, lr_matrices_test, device, padding)
        pred_matrices = [np.clip(p, 0.0, 1.0) for p in pred_matrices]

    submission_path = os.path.join("outputs", run_name, f"{run_name}_predictions.csv")
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df = save_submission_csv(pred_matrices, submission_path)

    plot_dir = os.path.join("outputs", run_name, "plots")
    if all_fold_metrics:
        plot_evaluation_barplots(all_fold_metrics, plot_dir)
    save_plots(all_preds, all_gt, submission_df["Predicted"].values, run_name, plot_dir)

    artifact = wandb.Artifact(
        name=f"submission-{run_name}"[:128],
        type="submission",
        description=f"GSRN — CV avg MAE: {avg_mae:.6f}",
    )
    artifact.add_file(submission_path)
    wandb.log_artifact(artifact)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _ensemble_predictions(fold_test_preds, n_test):
    n_models = len(fold_test_preds)
    print(f"\nEnsembling {n_models} models for submission")
    pred_matrices = []
    for i in range(n_test):
        avg_pred = np.mean([fold_test_preds[f][i] for f in range(n_models)], axis=0)
        pred_matrices.append(np.clip(avg_pred, 0.0, 1.0))
    return pred_matrices


def _full_retrain_gc(
    cfg,
    lr_matrices,
    hr_matrices,
    lr_matrices_test,
    train_feats,
    test_feats,
    device,
    global_step,
):
    print(
        f"\nFull retrain on all {lr_matrices.shape[0]} samples ({cfg.training.full_epochs} epochs)"
    )

    model = build_gc_model(cfg, device)
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.full_epochs, eta_min=cfg.training.eta_min
    )

    for epoch in range(cfg.training.full_epochs):
        train_mse, train_mae = train_one_epoch_gc(
            model, optimizer, lr_matrices, train_feats, hr_matrices, device, cfg
        )
        scheduler.step()

        global_step += 1
        wandb.log(
            {
                "full_retrain/epoch": epoch + 1,
                "full_retrain/mse": train_mse,
                "full_retrain/mae": train_mae,
            },
            step=global_step,
        )

        if (epoch + 1) % 10 == 0 or epoch == cfg.training.full_epochs - 1:
            print(
                f"Full retrain {epoch + 1:3d}/{cfg.training.full_epochs} | MSE: {train_mse:.6f} MAE: {train_mae:.6f}"
            )

    preds = predict_gc(model, lr_matrices_test, test_feats, device)
    return [np.clip(p, 0.0, 1.0) for p in preds]


def _rescale_predictions(pred_matrices, hr_matrices):
    hr_flat = hr_matrices.numpy().flatten()
    pred_flat = np.concatenate([p.flatten() for p in pred_matrices])
    for i in range(len(pred_matrices)):
        p = pred_matrices[i].flatten()
        rescaled = (p - pred_flat.mean()) / (
            pred_flat.std() + 1e-8
        ) * hr_flat.std() + hr_flat.mean()
        pred_matrices[i] = np.clip(rescaled.reshape(pred_matrices[i].shape), 0.0, 1.0)
    print(f"Rescaled predictions to match training HR distribution")
    return pred_matrices


def _print_cv_summary(seeds, n_splits, fold_mse, fold_mae):
    print(f"\n{'=' * 50}")
    print(f"{'Model':<12} {'MSE':>10} {'MAE':>10}")
    print(f"{'-' * 50}")
    model_i = 0
    for seed in seeds:
        for fold_i in range(n_splits):
            print(
                f"s{seed}/f{fold_i + 1:<3} {fold_mse[model_i]:>10.6f} {fold_mae[model_i]:>10.6f}"
            )
            model_i += 1
    print(f"{'-' * 50}")
    print(f"{'Avg':<12} {np.mean(fold_mse):>10.6f} {np.mean(fold_mae):>10.6f}")
    print(f"{'Std':<12} {np.std(fold_mse):>10.6f} {np.std(fold_mae):>10.6f}")
    print(f"{'=' * 50}")


def _print_eval_summary(all_fold_metrics):
    print(f"\n{'=' * 70}")
    print("Full Evaluation Metrics (per fold)")
    print(f"{'Fold':<8} {'MAE':>8} {'PCC':>8} {'JSD':>8} {'BC':>8} {'EC':>8} {'PC':>8}")
    print(f"{'-' * 70}")
    for i, fm in enumerate(all_fold_metrics):
        print(
            f"{'F' + str(i + 1):<8} {fm['MAE']:>8.5f} {fm['PCC']:>8.5f} {fm['JSD']:>8.5f} "
            f"{fm['BC']:>8.5f} {fm['EC']:>8.5f} {fm['PC']:>8.5f}"
        )
    avg = {k: np.mean([m[k] for m in all_fold_metrics]) for k in all_fold_metrics[0]}
    print(f"{'-' * 70}")
    print(
        f"{'Avg':<8} {avg['MAE']:>8.5f} {avg['PCC']:>8.5f} {avg['JSD']:>8.5f} "
        f"{avg['BC']:>8.5f} {avg['EC']:>8.5f} {avg['PC']:>8.5f}"
    )
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    run_name = build_run_name(cfg)

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    else:
        wandb.init(mode="disabled")

    lr_matrices, hr_matrices, lr_matrices_test = load_data(
        cfg.data.dir, cfg.data.lr_dim, cfg.data.hr_dim
    )
    print(
        f"Data loaded: {lr_matrices.shape[0]} train, {lr_matrices_test.shape[0]} test"
    )

    if cfg.model.name == "gsrn":
        run_gsrn(cfg, lr_matrices, hr_matrices, lr_matrices_test)
    else:
        run_graph_cent(cfg, lr_matrices, hr_matrices, lr_matrices_test)

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
