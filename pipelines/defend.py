import gc
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold

import wandb
from refine_u.evaluate import (
    compute_full_evaluation,
    mae_criterion,
    mse_criterion,
    save_submission_csv,
)
from refine_u.models.defend import (
    build_defend_model,
    create_dual_graph,
    create_dual_graph_feature_matrix,
    matrix_to_pyg,
    revert_dual,
)
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
)


# ---------------------------------------------------------------------------
# Train / evaluate / predict helpers
# ---------------------------------------------------------------------------


def _train_one_epoch(model, optimizer, source_pyg, target_mats, device, cfg, dual_edge_index=None):
    epoch_mse, epoch_mae = [], []
    batch_size = cfg.model.batch_size

    model.train()

    # Shuffle training data
    n = len(source_pyg)
    perm = torch.randperm(n)

    batch_counter = 0
    optimizer.zero_grad()

    for idx in perm:
        src_g = source_pyg[idx]
        tgt_m = target_mats[idx]

        if cfg.model.use_dual:
            pred = model(src_g, dual_edge_index)
            target = create_dual_graph_feature_matrix(tgt_m)
        else:
            pred = model(src_g)
            target = tgt_m

        loss = nn.L1Loss()(pred, target)
        loss.backward()

        batch_counter += 1

        # Mini-batch gradient accumulation
        if batch_counter % batch_size == 0 or batch_counter == n:
            optimizer.step()
            optimizer.zero_grad()

        # Track metrics on the predicted matrix (not dual)
        with torch.no_grad():
            if cfg.model.use_dual:
                pred_mat = revert_dual(pred.detach(), cfg.data.hr_dim)
            else:
                pred_mat = pred.detach()
            epoch_mse.append(mse_criterion(pred_mat, tgt_m).item())
            epoch_mae.append(mae_criterion(pred_mat, tgt_m).item())

    torch.cuda.empty_cache()
    gc.collect()

    return np.mean(epoch_mse), np.mean(epoch_mae)


def _evaluate(model, source_pyg, target_mats, device, cfg, dual_edge_index=None):
    mse_error, mae_error = [], []

    model.eval()
    with torch.no_grad():
        for src_g, tgt_m in zip(source_pyg, target_mats):
            if cfg.model.use_dual:
                pred = model(src_g, dual_edge_index)
                pred_mat = revert_dual(pred, cfg.data.hr_dim)
            else:
                pred_mat = model(src_g)

            mse_error.append(mse_criterion(pred_mat, tgt_m).item())
            mae_error.append(mae_criterion(pred_mat, tgt_m).item())

    return np.mean(mse_error), np.mean(mae_error)


def _predict(model, source_pyg, device, cfg, dual_edge_index=None):
    preds = []
    model.eval()
    with torch.no_grad():
        for src_g in source_pyg:
            if cfg.model.use_dual:
                pred = model(src_g, dual_edge_index)
                pred_mat = revert_dual(pred, cfg.data.hr_dim)
            else:
                pred_mat = model(src_g)
            preds.append(np.clip(pred_mat.cpu().numpy(), 0.0, 1.0))
    return preds


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_defend(cfg, lr_matrices, hr_matrices, lr_matrices_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert all matrices to PyG graphs
    node_feat_init = cfg.model.node_feat_init
    node_feat_dim = cfg.model.node_feat_dim

    print("Converting LR matrices to PyG graphs...")
    source_pyg_all = [
        matrix_to_pyg(lr_matrices[i].to(device), node_feat_init, node_feat_dim)
        for i in range(lr_matrices.shape[0])
    ]
    target_mat_all = [hr_matrices[i].to(device) for i in range(hr_matrices.shape[0])]

    print("Converting test LR matrices to PyG graphs...")
    test_pyg_all = [
        matrix_to_pyg(lr_matrices_test[i].to(device), node_feat_init, node_feat_dim)
        for i in range(lr_matrices_test.shape[0])
    ]

    # Dual graph setup
    dual_edge_index = None
    if cfg.model.use_dual:
        n_target = cfg.data.hr_dim
        dual_domain = torch.ones((n_target, n_target), dtype=torch.float, device=device)
        dual_edge_index, _ = create_dual_graph(dual_domain)

    seeds = list(cfg.training.seeds)
    total_models = len(seeds) * cfg.training.splits

    print(f"\n{'=' * 60}")
    print(f"DEFEND ({cfg.model.sr_method}) — {cfg.training.splits}-fold CV x {len(seeds)} seeds = {total_models} models")
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
        indices = np.arange(len(source_pyg_all))

        for fold_idx, (train_index, test_index) in enumerate(cv.split(indices)):
            model_idx += 1
            print(f"\n--- Seed {seed}, Fold {fold_idx + 1}/{cfg.training.splits} (model {model_idx}/{total_models}) ---")

            model = build_defend_model(cfg, device)

            if model_idx == 1:
                print(model)
                n_params = sum(p.numel() for p in model.parameters())
                print(f"Total parameters: {n_params / 1e6:.4f}M")
                wandb.config.update({"n_params": n_params})

            optimizer = optim.Adam(
                model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
            )

            src_train = [source_pyg_all[i] for i in train_index]
            tgt_train = [target_mat_all[i] for i in train_index]
            src_val = [source_pyg_all[i] for i in test_index]
            tgt_val = [target_mat_all[i] for i in test_index]

            best_val_mae = float("inf")
            best_state = None
            wait = 0

            for epoch in range(cfg.training.epochs):
                train_mse, train_mae = _train_one_epoch(
                    model, optimizer, src_train, tgt_train, device, cfg, dual_edge_index
                )
                val_mse, val_mae = _evaluate(
                    model, src_val, tgt_val, device, cfg, dual_edge_index
                )

                # Early stopping / best checkpoint
                if cfg.training.best_ckpt and val_mae < best_val_mae - 1e-5:
                    best_val_mae = val_mae
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    wait = 0
                elif cfg.training.patience > 0 and epoch >= cfg.model.get("warm_up_epochs", 0):
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
                    },
                )

                if (epoch + 1) % 10 == 0 or epoch == cfg.training.epochs - 1:
                    print(
                        f"Epoch {epoch + 1:3d} | Train MSE: {train_mse:.6f} MAE: {train_mae:.6f} "
                        f"| Val MSE: {val_mse:.6f} MAE: {val_mae:.6f}"
                    )

            if cfg.training.best_ckpt and best_state is not None:
                model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
                print(f"  Restored best checkpoint (val MAE: {best_val_mae:.6f})")

            val_mse, val_mae = _evaluate(model, src_val, tgt_val, device, cfg, dual_edge_index)
            fold_mse.append(val_mse)
            fold_mae.append(val_mae)

            fold_key = f"s{seed}_f{fold_idx + 1}" if len(seeds) > 1 else f"fold_{fold_idx + 1}"
            wandb.log({f"{fold_key}/val_mse": val_mse, f"{fold_key}/val_mae": val_mae})

            # Collect CV predictions
            fold_pred_matrices, fold_gt_matrices = [], []
            model.eval()
            with torch.no_grad():
                for src_g, tgt_m in zip(src_val, tgt_val):
                    if cfg.model.use_dual:
                        pred = model(src_g, dual_edge_index)
                        pred_mat = revert_dual(pred, cfg.data.hr_dim)
                    else:
                        pred_mat = model(src_g)
                    fold_pred_matrices.append(np.clip(pred_mat.cpu().numpy(), 0.0, 1.0))
                    fold_gt_matrices.append(tgt_m.cpu().numpy())

            fold_pred_matrices = np.array(fold_pred_matrices)
            fold_gt_matrices = np.array(fold_gt_matrices)
            all_fold_pred_matrices.append((fold_pred_matrices, fold_gt_matrices))

            print(f"Seed {seed}, Fold {fold_idx + 1} — MSE: {val_mse:.6f} MAE: {val_mae:.6f}")

            # Test predictions for ensemble
            fold_test = None
            if cfg.training.ensemble:
                fold_test = _predict(model, test_pyg_all, device, cfg, dual_edge_index)
                fold_test_preds.append(fold_test)

            # Per-fold plots
            if cfg.training.plots:
                log_fold_plots(model_idx, fold_pred_matrices, fold_gt_matrices, fold_test)

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

    # MAE/MSE bar plot
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
        fold_csv_path = os.path.join(fold_output_dir, f"predictions_fold_{fold_num}.csv")
        save_submission_csv(fold_preds, fold_csv_path)
        fold_artifact.add_file(fold_csv_path, name=f"predictions_fold_{fold_num}.csv")

    wandb.log_artifact(fold_artifact)

    # Full evaluation metrics (optional)
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
        avg_metrics = {k: np.mean([m[k] for m in all_fold_metrics]) for k in all_fold_metrics[0]}
        wandb.log({f"cv/avg_{k.lower()}": v for k, v in avg_metrics.items()})

    # Phase 2: Generate submission
    if cfg.training.ensemble:
        pred_matrices = ensemble_predictions(fold_test_preds, lr_matrices_test.shape[0])
    else:
        pred_matrices = _predict(model, test_pyg_all, device, cfg, dual_edge_index)

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
        description=f"DEFEND {cfg.model.sr_method} — CV avg MAE: {avg_mae:.6f}",
    )
    artifact.add_file(submission_path)
    wandb.log_artifact(artifact)
