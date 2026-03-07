import os

import networkx as nx
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

from refine_u.project_original_files.MatrixVectorizer import MatrixVectorizer

mse_criterion = nn.MSELoss()
mae_criterion = nn.L1Loss()


def compute_loss(pred, target, loss_type="mse"):
    mse = mse_criterion(pred, target)
    if loss_type == "mse":
        return mse
    mae = mae_criterion(pred, target)
    if loss_type == "l1":
        return mae
    return 0.5 * mse + 0.5 * mae


def compute_full_evaluation(pred_matrices, gt_matrices):
    """Compute all 6 evaluation metrics: MAE, PCC, JSD, BC, EC, PC."""
    num_samples = len(pred_matrices)
    mae_bc, mae_ec, mae_pc = [], [], []
    pred_1d_list, gt_1d_list = [], []

    for i in range(num_samples):
        pred_mat = pred_matrices[i]
        gt_mat = gt_matrices[i]

        pred_graph = nx.from_numpy_array(pred_mat, edge_attr="weight")
        gt_graph = nx.from_numpy_array(gt_mat, edge_attr="weight")

        pred_bc = list(nx.betweenness_centrality(pred_graph, weight="weight").values())
        gt_bc = list(nx.betweenness_centrality(gt_graph, weight="weight").values())

        try:
            pred_ec = list(
                nx.eigenvector_centrality(pred_graph, weight="weight").values()
            )
        except nx.PowerIterationFailedConvergence:
            pred_ec = list(
                nx.eigenvector_centrality(
                    pred_graph, weight="weight", max_iter=1000
                ).values()
            )
        try:
            gt_ec = list(nx.eigenvector_centrality(gt_graph, weight="weight").values())
        except nx.PowerIterationFailedConvergence:
            gt_ec = list(
                nx.eigenvector_centrality(
                    gt_graph, weight="weight", max_iter=1000
                ).values()
            )

        pred_pc = list(nx.pagerank(pred_graph, weight="weight").values())
        gt_pc = list(nx.pagerank(gt_graph, weight="weight").values())

        mae_bc.append(mean_absolute_error(pred_bc, gt_bc))
        mae_ec.append(mean_absolute_error(pred_ec, gt_ec))
        mae_pc.append(mean_absolute_error(pred_pc, gt_pc))

        pred_1d_list.append(MatrixVectorizer.vectorize(pred_mat))
        gt_1d_list.append(MatrixVectorizer.vectorize(gt_mat))

    pred_1d = np.concatenate(pred_1d_list)
    gt_1d = np.concatenate(gt_1d_list)

    return {
        "MAE": mean_absolute_error(pred_1d, gt_1d),
        "PCC": pearsonr(pred_1d, gt_1d)[0],
        "JSD": jensenshannon(pred_1d, gt_1d),
        "BC": np.mean(mae_bc),
        "EC": np.mean(mae_ec),
        "PC": np.mean(mae_pc),
    }


def save_submission_csv(pred_matrices, path):
    """Save predictions as CSV in Kaggle submission format."""
    preds_vect = []
    for pred in pred_matrices:
        pred_clipped = np.clip(pred, 0.0, 1.0)
        pred_vect = MatrixVectorizer.vectorize(pred_clipped, include_diagonal=False)
        preds_vect.append(pred_vect)
    preds_flat = np.concatenate(preds_vect)
    ids = np.arange(1, len(preds_flat) + 1)
    df = pd.DataFrame({"ID": ids, "Predicted": preds_flat})
    df.to_csv(path, index=False)
    print(f"Saved submission to {path} ({len(df)} rows)")
    return df
