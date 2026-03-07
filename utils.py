import numpy as np


def build_run_name(cfg):
    parts = []

    if cfg.model.name == "graph_cent":
        parts.append(
            "gcn" if cfg.model.gcn_type == "traditional" else cfg.model.gcn_type
        )
        parts.append(cfg.model.feat_type)
        if cfg.model.gcn_type == "unet" and cfg.model.pool_ratios:
            pool_str = "_".join(str(r) for r in cfg.model.pool_ratios)
            parts.append(f"p{pool_str}")
        if cfg.model.gated_skip:
            parts.append("gated")
        if cfg.model.low_rank_k > 0:
            parts.append(f"lr{cfg.model.low_rank_k}")
    elif cfg.model.name == "gsrn":
        parts.append("gsrn")
        parts.append(f"lmbda{cfg.model.lmbda}")

    parts.append(f"{cfg.training.epochs}ep")
    parts.append(f"{cfg.training.splits}f")

    seeds = list(cfg.training.seeds)
    if len(seeds) > 1:
        parts.append(f"{len(seeds)}s")

    parts.append(cfg.training.loss)
    parts.append("ens" if cfg.training.ensemble else "full")

    return "-".join(parts)


def ensemble_predictions(fold_test_preds, n_test):
    n_models = len(fold_test_preds)
    print(f"\nEnsembling {n_models} models for submission")
    pred_matrices = []
    for i in range(n_test):
        avg_pred = np.mean([fold_test_preds[f][i] for f in range(n_models)], axis=0)
        pred_matrices.append(np.clip(avg_pred, 0.0, 1.0))
    return pred_matrices


def rescale_predictions(pred_matrices, hr_matrices):
    hr_flat = hr_matrices.numpy().flatten()
    pred_flat = np.concatenate([p.flatten() for p in pred_matrices])
    for i in range(len(pred_matrices)):
        p = pred_matrices[i].flatten()
        rescaled = (p - pred_flat.mean()) / (
            pred_flat.std() + 1e-8
        ) * hr_flat.std() + hr_flat.mean()
        pred_matrices[i] = np.clip(rescaled.reshape(pred_matrices[i].shape), 0.0, 1.0)
    print("Rescaled predictions to match training HR distribution")
    return pred_matrices


def print_cv_summary(seeds, n_splits, fold_mse, fold_mae):
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


def print_eval_summary(all_fold_metrics):
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
