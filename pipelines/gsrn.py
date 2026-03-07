import os

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import KFold

import wandb
from refine_u.evaluate import (
    compute_full_evaluation,
    mae_criterion,
    mse_criterion,
    save_submission_csv,
)
from refine_u.models import GSRNet
from refine_u.models.gsrn import pad_HR_adj, unpad
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
# Model builder
# ---------------------------------------------------------------------------


def _build_model(cfg, device):
    ks = list(cfg.model.pool_ratios)
    return GSRNet(
        ks=ks,
        lr_dim=cfg.data.lr_dim,
        hr_dim=cfg.data.hr_dim,
        hidden_dim=cfg.model.hidden_dim,
        padding=cfg.model.padding,
    ).to(device)


# ---------------------------------------------------------------------------
# Train / evaluate / predict
# ---------------------------------------------------------------------------


def _train_one_epoch(model, optimizer, lr_data, hr_data, device, cfg):
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


def _evaluate(model, lr_data, hr_data, device, padding):
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


def _predict(model, lr_data, device, padding):
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
# Main pipeline
# ---------------------------------------------------------------------------


def run_gsrn(cfg, lr_matrices, hr_matrices, lr_matrices_test):
    # GSRN uses plain Python lists (not nn.ModuleList), so .to(device) doesn't propagate
    device = torch.device("cpu")
    print(f"Using device: {device} (forced CPU for GSRN)")

    seeds = list(cfg.training.seeds)
    padding = cfg.model.padding

    fold_mse, fold_mae = [], []
    fold_test_preds = []
    all_fold_pred_matrices = []
    global_step = 0
    model_idx = 0

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # GSRN shares a single model across folds (matches original)
        model = _build_model(cfg, device)
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
            model_idx += 1
            print(f"\n--- Seed {seed}, Fold {fold_idx + 1}/{cfg.training.splits} ---")

            x_train = lr_matrices[train_index]
            y_train = hr_matrices[train_index]
            x_val = lr_matrices[test_index]
            y_val = hr_matrices[test_index]

            for epoch in range(cfg.training.epochs):
                train_mse, train_mae = _train_one_epoch(
                    model, optimizer, x_train, y_train, device, cfg
                )
                val_mse, val_mae = _evaluate(model, x_val, y_val, device, padding)

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
                )

                if (epoch + 1) % 10 == 0 or epoch == cfg.training.epochs - 1:
                    print(
                        f"Epoch {epoch + 1:3d} | Train MSE: {train_mse:.6f} MAE: {train_mae:.6f} "
                        f"| Val MSE: {val_mse:.6f} MAE: {val_mae:.6f}"
                    )

            val_mse, val_mae = _evaluate(model, x_val, y_val, device, padding)
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

            fold_pred_matrices = np.array(fold_pred_matrices)
            fold_gt_matrices = np.array(fold_gt_matrices)
            all_fold_pred_matrices.append((fold_pred_matrices, fold_gt_matrices))

            print(f"Fold {fold_idx + 1} — MSE: {val_mse:.6f} MAE: {val_mae:.6f}")

            # Test predictions for ensemble
            fold_test = None
            if cfg.training.ensemble:
                fold_test = _predict(model, lr_matrices_test, device, padding)
                fold_test_preds.append(fold_test)

            # Per-fold plots
            if cfg.training.plots:
                log_fold_plots(model_idx, fold_pred_matrices, fold_gt_matrices, fold_test)

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
        save_submission_csv(fold_preds, fold_csv_path)
        fold_artifact.add_file(fold_csv_path, name=f"predictions_fold_{fold_num}.csv")

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

    if cfg.training.ensemble:
        pred_matrices = ensemble_predictions(fold_test_preds, lr_matrices_test.shape[0])
    else:
        pred_matrices = _predict(model, lr_matrices_test, device, padding)
        pred_matrices = [np.clip(p, 0.0, 1.0) for p in pred_matrices]

    submission_path = os.path.join("outputs", run_name, f"{run_name}_predictions.csv")
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df = save_submission_csv(pred_matrices, submission_path)

    if cfg.training.plots:
        if all_fold_metrics:
            plot_evaluation_barplots(all_fold_metrics)
        log_average_plots(all_fold_pred_matrices, fold_test_preds or None)

    artifact = wandb.Artifact(
        name=f"submission-{run_name}"[:128],
        type="submission",
        description=f"GSRN — CV avg MAE: {avg_mae:.6f}",
    )
    artifact.add_file(submission_path)
    wandb.log_artifact(artifact)
