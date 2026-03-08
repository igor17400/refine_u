import copy
import os

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import KFold

import wandb
from refine_u.data import GraphCentralityDataset, augment_adjacency
from refine_u.evaluate import (
    compute_full_evaluation,
    compute_loss,
    compute_unet_loss,
    mae_criterion,
    mse_criterion,
    save_submission_csv,
)
from refine_u.models import GraphCentMapperModel
from refine_u.plotting import (
    log_average_plots,
    log_fold_plots,
    log_mae_mse_barplot,
    plot_evaluation_barplots,
)
from refine_u.utils import (
    build_run_name,
    ensemble_predictions,
    print_cv_summary,
    print_eval_summary,
    rescale_predictions,
)

# ---------------------------------------------------------------------------
# Model / feature builders
# ---------------------------------------------------------------------------


def _build_model(cfg, device):
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
        output_activation=cfg.model.output_activation,
    ).to(device)


def _get_features(cfg, lr_matrices, hr_matrices, lr_matrices_test):
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
# Train / evaluate / predict
# ---------------------------------------------------------------------------


def _train_one_epoch(model, optimizer, adj_data, feat_data, labels, device, cfg):
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
        unet_loss = compute_unet_loss(x_in, x_out, cfg.model.unet_loss_type)
        loss = pred_loss + (
            cfg.model.unet_loss_weight * unet_loss if model.gcn_type == "unet" else 0.0
        )

        loss.backward()
        optimizer.step()

        epoch_mse.append(mse_criterion(pred, hr).item())
        epoch_mae.append(mae_criterion(pred, hr).item())

    return np.mean(epoch_mse), np.mean(epoch_mae)


def _evaluate(model, adj_data, feat_data, labels, device):
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


def _predict(model, adj_data, feat_data, device):
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
# Full retrain (non-ensemble path)
# ---------------------------------------------------------------------------


def _full_retrain(
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

    model = _build_model(cfg, device)
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.full_epochs, eta_min=cfg.training.eta_min
    )

    for epoch in range(cfg.training.full_epochs):
        train_mse, train_mae = _train_one_epoch(
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
        )

        if (epoch + 1) % 10 == 0 or epoch == cfg.training.full_epochs - 1:
            print(
                f"Full retrain {epoch + 1:3d}/{cfg.training.full_epochs} | MSE: {train_mse:.6f} MAE: {train_mae:.6f}"
            )

    preds = _predict(model, lr_matrices_test, test_feats, device)
    return [np.clip(p, 0.0, 1.0) for p in preds]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_refine_u(cfg, lr_matrices, hr_matrices, lr_matrices_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_feats, test_feats = _get_features(
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
    fold_test_preds = []
    all_fold_pred_matrices = []
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

            model = _build_model(cfg, device)

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
                train_mse, train_mae = _train_one_epoch(
                    model, optimizer, x_train, f_train, y_train, device, cfg
                )
                val_mse, val_mae = _evaluate(model, x_val, f_val, y_val, device)
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
                )

                if (epoch + 1) % 10 == 0 or epoch == cfg.training.epochs - 1:
                    print(
                        f"Epoch {epoch + 1:3d} | Train MSE: {train_mse:.6f} MAE: {train_mae:.6f} "
                        f"| Val MSE: {val_mse:.6f} MAE: {val_mae:.6f}"
                    )

            if cfg.training.best_ckpt and best_state is not None:
                model.load_state_dict(best_state)
                print(f"  Restored best checkpoint (val MAE: {best_val_mae:.6f})")

            val_mse, val_mae = _evaluate(model, x_val, f_val, y_val, device)
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

            fold_pred_matrices = np.array(fold_pred_matrices)
            fold_gt_matrices = np.array(fold_gt_matrices)
            all_fold_pred_matrices.append((fold_pred_matrices, fold_gt_matrices))

            print(
                f"Seed {seed}, Fold {fold_idx + 1} — MSE: {val_mse:.6f} MAE: {val_mae:.6f}"
            )

            # Test predictions for ensemble
            fold_test = None
            if cfg.training.ensemble:
                fold_test = _predict(model, lr_matrices_test, test_feats, device)
                fold_test_preds.append(fold_test)

            # Per-fold plots
            if cfg.training.plots:
                log_fold_plots(
                    model_idx, fold_pred_matrices, fold_gt_matrices, fold_test
                )

    # CV summary
    print_cv_summary(seeds, cfg.training.splits, fold_mse, fold_mae)
    avg_mae = np.mean(fold_mae)
    wandb.log(
        {
            "cv/avg_mse": np.mean(fold_mse),
            "cv/avg_mae": avg_mae,
            "cv/std_mse": np.std(fold_mse),
            "cv/std_mae": np.std(fold_mae),
        }
    )

    # MAE/MSE bar plot (always logged)
    log_mae_mse_barplot(fold_mse, fold_mae)

    run_name = build_run_name(cfg)

    # Save per-fold prediction CSVs
    print(f"\n{'=' * 60}")
    print("Post-training: saving fold CSVs")
    print(f"{'=' * 60}")

    fold_output_dir = os.path.join("outputs", run_name, "folds")
    os.makedirs(fold_output_dir, exist_ok=True)

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
        gt_csv_path = os.path.join(
            fold_output_dir, f"groundtruth_fold_{fold_num}.csv"
        )
        save_submission_csv(fold_preds, fold_csv_path)
        save_submission_csv(fold_gts, gt_csv_path)
        fold_artifact.add_file(fold_csv_path, name=f"predictions_fold_{fold_num}.csv")
        fold_artifact.add_file(gt_csv_path, name=f"groundtruth_fold_{fold_num}.csv")

    wandb.log_artifact(fold_artifact)

    # Full evaluation metrics (optional — slow due to NetworkX centrality)
    all_fold_metrics = []
    if cfg.training.eval_metrics:
        print("Computing full evaluation metrics...")
        for fold_i, (fold_preds, fold_gts) in enumerate(all_fold_pred_matrices):
            fold_metrics = compute_full_evaluation(fold_preds, fold_gts)
            all_fold_metrics.append(fold_metrics)
            print(
                f"  Fold {fold_i + 1}: MAE={fold_metrics['MAE']:.5f} PCC={fold_metrics['PCC']:.5f} "
                f"JSD={fold_metrics['JSD']:.5f} BC={fold_metrics['BC']:.5f} "
                f"EC={fold_metrics['EC']:.5f} PC={fold_metrics['PC']:.5f}"
            )
        print_eval_summary(all_fold_metrics)
        avg_metrics = {
            k: np.mean([m[k] for m in all_fold_metrics]) for k in all_fold_metrics[0]
        }
        wandb.log({f"cv/avg_{k.lower()}": v for k, v in avg_metrics.items()})

    # Phase 2: Generate submission
    if cfg.training.ensemble:
        pred_matrices = ensemble_predictions(fold_test_preds, lr_matrices_test.shape[0])
    else:
        pred_matrices = _full_retrain(
            cfg,
            lr_matrices,
            hr_matrices,
            lr_matrices_test,
            train_feats,
            test_feats,
            device,
            global_step,
        )

    if cfg.training.rescale:
        pred_matrices = rescale_predictions(pred_matrices, hr_matrices)

    submission_path = os.path.join("outputs", run_name, f"{run_name}_predictions.csv")
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df = save_submission_csv(pred_matrices, submission_path)

    # Plots
    if cfg.training.plots:
        if all_fold_metrics:
            plot_evaluation_barplots(all_fold_metrics)
        log_average_plots(all_fold_pred_matrices, fold_test_preds or None)

    # Wandb artifact
    artifact = wandb.Artifact(
        name=f"submission-{run_name}"[:128],
        type="submission",
        description=f"CV avg MAE: {avg_mae:.6f}",
    )
    artifact.add_file(submission_path)
    wandb.log_artifact(artifact)
