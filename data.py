import os

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from refine_u.project_original_files.MatrixVectorizer import MatrixVectorizer


def load_data(data_dir: str, lr_dim: int, hr_dim: int):
    lr_train = pd.read_csv(os.path.join(data_dir, "lr_train.csv")).values
    hr_train = pd.read_csv(os.path.join(data_dir, "hr_train.csv")).values
    lr_test = pd.read_csv(os.path.join(data_dir, "lr_test.csv")).values

    LR_TRAIN = torch.zeros((lr_train.shape[0], lr_dim, lr_dim))
    HR_TRAIN = torch.zeros((hr_train.shape[0], hr_dim, hr_dim))
    LR_TEST = torch.zeros((lr_test.shape[0], lr_dim, lr_dim))

    for i in range(lr_train.shape[0]):
        LR_TRAIN[i] = torch.from_numpy(
            MatrixVectorizer.anti_vectorize(lr_train[i], lr_dim)
        ).float()
    for i in range(hr_train.shape[0]):
        HR_TRAIN[i] = torch.from_numpy(
            MatrixVectorizer.anti_vectorize(hr_train[i], hr_dim)
        ).float()
    for i in range(lr_test.shape[0]):
        LR_TEST[i] = torch.from_numpy(
            MatrixVectorizer.anti_vectorize(lr_test[i], lr_dim)
        ).float()

    return LR_TRAIN, HR_TRAIN, LR_TEST


def augment_adjacency(adj, noise_std=0.0, edge_drop=0.0):
    aug = adj.clone()
    if noise_std > 0:
        aug = aug + torch.randn_like(aug) * noise_std
        aug = aug.clamp(0.0, 1.0)
        aug = (aug + aug.transpose(-2, -1)) / 2.0
    if edge_drop > 0:
        mask = torch.rand_like(aug) > edge_drop
        mask = mask & mask.transpose(-2, -1)
        aug = aug * mask.float()
    return aug


class GraphCentralityDataset(Dataset):
    def __init__(self, adjacency_matrices, targets=None, save_path=None):
        self.adj_matrices = adjacency_matrices
        self.targets = targets
        self.num_samples, self.num_nodes, _ = self.adj_matrices.shape
        self.b_features = None
        self.e_features = None
        self.save_path = save_path

        if save_path:
            b_path = save_path + "_b_features.pt"
            e_path = save_path + "_e_features.pt"
            if os.path.exists(b_path) and os.path.exists(e_path):
                print(f"Loading precomputed features from {save_path}...")
                self.b_features = torch.load(b_path)
                self.e_features = torch.load(e_path)
                return
            else:
                print(f"No precomputed features found at {save_path}. Computing...")

        print(f"Precomputing betweenness centrality for {self.num_samples} graphs...")
        self.b_features = self._precompute_all_centrality()
        if save_path:
            torch.save(self.b_features, save_path + "_b_features.pt")

        print(f"Precomputing eigenvalue embeddings for {self.num_samples} graphs...")
        self.e_features = self._precompute_all_eigenvalue_embeddings()
        if save_path:
            torch.save(self.e_features, save_path + "_e_features.pt")

    def _precompute_all_eigenvalue_embeddings(self):
        e_features = torch.zeros((self.num_samples, self.num_nodes, 16))
        for i in range(self.num_samples):
            e_features[i] = self._calculate_eigenvalue_embedding(self.adj_matrices[i])
        return e_features

    def _precompute_all_centrality(self):
        b_features = torch.zeros((self.num_samples, self.num_nodes, 1))
        for i in range(self.num_samples):
            b_features[i] = self._calculate_single_betweenness(
                self.adj_matrices[i].numpy()
            )
        return b_features

    def _calculate_single_betweenness(self, adj_matrix):
        G = nx.Graph()
        G.add_nodes_from(range(self.num_nodes))

        rows, cols = np.where(adj_matrix > 0)
        for r, c in zip(rows, cols):
            if r < c:
                G.add_edge(r, c, weight=1.0 / adj_matrix[r, c])

        b_dict = nx.betweenness_centrality(G, weight="weight")
        b_array = np.array([b_dict[n] for n in range(self.num_nodes)])
        return torch.tensor(b_array, dtype=torch.float32).unsqueeze(-1)

    def _calculate_eigenvalue_embedding(self, adj_matrix: torch.Tensor, k: int = 16):
        G = nx.from_numpy_array(adj_matrix.cpu().numpy())
        normalized_laplacian = nx.normalized_laplacian_matrix(G)
        L = torch.tensor(normalized_laplacian.todense(), dtype=torch.float)
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        initial_embeddings = eigenvectors[:, 1 : k + 1]
        return initial_embeddings

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        adj = self.adj_matrices[idx]
        if self.b_features is None or self.e_features is None:
            raise ValueError("Features have not been precomputed.")
        b_feat = self.b_features[idx]
        e_feat = self.e_features[idx]
        feat = e_feat
        if self.targets is not None:
            return adj, feat, self.targets[idx]
        else:
            return adj, feat
