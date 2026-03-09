"""
DEFEND: Graph super-resolution via TransformerConv.

Three SR methods:
  - LA (Linear Algebraic): TransformerConv on source graph, dot product for target adjacency
  - BiLC (Bipartite Linear Combination): learned linear mapping from source to target space
  - BiMP (Bipartite Message Passing): bipartite graph attention for source-to-target mapping

Optionally wraps in a DualModel that operates on the line graph (dual) of the target.

Adapted from: https://github.com/brain_graph_challenge/src/defend
"""

import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GraphNorm, TransformerConv
from torch_geometric.utils import dense_to_sparse


# ---------------------------------------------------------------------------
# PyG graph conversion utilities
# ---------------------------------------------------------------------------


def matrix_to_pyg(adj_matrix, node_feature_init="adj", node_feat_dim=1):
    """Convert an adjacency matrix tensor to a PyG Data object."""
    n_nodes = adj_matrix.shape[0]

    # Edge features: flatten adjacency
    edge_attr = adj_matrix.reshape(-1, 1)

    # Node features
    if node_feature_init == "adj":
        node_feat = adj_matrix
    elif node_feature_init == "random":
        node_feat = torch.randn(n_nodes, node_feat_dim, device=adj_matrix.device)
    elif node_feature_init == "ones":
        node_feat = torch.ones(n_nodes, node_feat_dim, device=adj_matrix.device)
    elif node_feature_init == "identity":
        node_feat = torch.eye(n_nodes, device=adj_matrix.device)
    else:
        raise ValueError(f"Unsupported node feature init: {node_feature_init}")

    rows, cols = torch.meshgrid(
        torch.arange(n_nodes), torch.arange(n_nodes), indexing="ij"
    )
    pos_edge_index = torch.stack([rows.flatten(), cols.flatten()], dim=0).to(
        adj_matrix.device
    )

    return Data(x=node_feat, pos_edge_index=pos_edge_index, edge_attr=edge_attr)


# ---------------------------------------------------------------------------
# Dual graph utilities
# ---------------------------------------------------------------------------


def create_dual_graph(adjacency_matrix):
    """Returns (edge_index, node_feature_matrix) for the undirected dual graph."""
    n = adjacency_matrix.shape[0]
    row, col = torch.triu_indices(n, n, offset=1)
    all_edges = torch.stack([row, col], dim=1).to(adjacency_matrix.device)
    actual_edges_mask = adjacency_matrix[row, col].nonzero().view(-1)
    actual_edges = all_edges[actual_edges_mask]

    max_possible_edges = row.size(0)

    edge_to_nodes = torch.zeros(
        (max_possible_edges, n), dtype=torch.float, device=adjacency_matrix.device
    )
    edge_to_nodes[actual_edges_mask, actual_edges[:, 0]] = 1.0
    edge_to_nodes[actual_edges_mask, actual_edges[:, 1]] = 1.0

    shared_nodes_matrix = edge_to_nodes @ edge_to_nodes.t()
    shared_nodes_matrix.fill_diagonal_(0)

    edge_index = shared_nodes_matrix.nonzero(as_tuple=False).t().contiguous()

    node_feat_matrix = torch.zeros(
        (max_possible_edges, 1), dtype=torch.float, device=adjacency_matrix.device
    )
    node_feat_matrix[actual_edges_mask] = (
        adjacency_matrix[actual_edges[:, 0], actual_edges[:, 1]].view(-1, 1).float()
    )

    torch.cuda.empty_cache()
    gc.collect()

    return edge_index, node_feat_matrix


def create_dual_graph_feature_matrix(adjacency_matrix):
    """Returns node_feature_matrix for the dual graph."""
    n = adjacency_matrix.shape[0]
    row, col = torch.triu_indices(n, n, offset=1)
    actual_edges_mask = adjacency_matrix[row, col].nonzero().view(-1).cpu()

    node_feat_matrix = torch.zeros(
        (row.size(0), 1), dtype=torch.float, device=adjacency_matrix.device
    )
    node_feat_matrix[actual_edges_mask] = (
        adjacency_matrix[row[actual_edges_mask], col[actual_edges_mask]]
        .view(-1, 1)
        .float()
    )
    return node_feat_matrix


def revert_dual(node_feat, n_nodes):
    """Reverts dual node features back to an adjacency matrix."""
    adj = torch.zeros((n_nodes, n_nodes), dtype=torch.float, device=node_feat.device)
    row, col = torch.triu_indices(n_nodes, n_nodes, offset=1)
    adj[row, col] = node_feat.view(-1)
    adj[col, row] = node_feat.view(-1)
    return adj


# ---------------------------------------------------------------------------
# SR Models
# ---------------------------------------------------------------------------


class LA(nn.Module):
    """TransformerConv-based Linear Algebraic SR."""

    def __init__(
        self,
        n_source_nodes,
        n_target_nodes,
        num_heads=4,
        edge_dim=1,
        dropout=0.2,
        beta=False,
        min_max_scale=True,
        binarize=False,
    ):
        super().__init__()
        assert n_target_nodes % num_heads == 0

        self.min_max_scale = min_max_scale
        self.binarize = binarize

        self.conv1 = TransformerConv(
            n_source_nodes,
            n_target_nodes // num_heads,
            heads=num_heads,
            edge_dim=edge_dim,
            dropout=dropout,
            beta=beta,
        )
        self.bn1 = GraphNorm(n_target_nodes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)  # (n_s, n_t)

        # Dot product for target adjacency
        xt = x.T @ x  # (n_t, n_t)

        if self.min_max_scale:
            xt_min = torch.min(xt)
            xt_max = torch.max(xt)
            xt = (xt - xt_min) / (xt_max - xt_min + 1e-8)

        if self.binarize:
            xt = (xt - torch.mean(xt)) / torch.std(xt)
            xt = torch.sigmoid(xt)

        return xt


class BiLC(nn.Module):
    """Bipartite Linear Combination SR."""

    def __init__(
        self,
        n_source_nodes,
        n_target_nodes,
        num_heads=4,
        edge_dim=1,
        dropout=0.2,
        beta=False,
        hidden_dim=32,
        refine_target=False,
        target_refine_fully_connected=False,
        min_max_scale=True,
        binarize=False,
    ):
        super().__init__()
        self.n_target_nodes = n_target_nodes
        self.min_max_scale = min_max_scale
        self.binarize = binarize

        self.conv1 = TransformerConv(
            n_source_nodes,
            hidden_dim,
            heads=num_heads,
            edge_dim=edge_dim,
            dropout=dropout,
            beta=beta,
        )
        self.bn1 = GraphNorm(hidden_dim * num_heads)

        self.bipartitie_fc = nn.Linear(n_source_nodes, n_target_nodes)

        self.refine_target = refine_target
        self.target_refine_fully_connected = target_refine_fully_connected
        if refine_target:
            if not target_refine_fully_connected:
                self.target_adj_fc = nn.Linear(n_source_nodes, n_target_nodes)
                edge_dim = 1
            else:
                edge_dim = None

            self.conv2 = TransformerConv(
                hidden_dim * num_heads,
                hidden_dim,
                heads=num_heads,
                edge_dim=edge_dim,
                dropout=dropout,
                beta=beta,
            )
            self.bn2 = GraphNorm(hidden_dim * num_heads)

    def _fully_connected_edge_index(self, device):
        n = self.n_target_nodes
        row = torch.arange(n).repeat(n)
        col = torch.arange(n).repeat_interleave(n)
        return torch.stack([row, col], dim=0).to(device)

    def linear_sr(self, x):
        xt = self.bipartitie_fc(x.T).T  # (n_t, h)

        if self.refine_target:
            if self.target_refine_fully_connected:
                edge_index = self._fully_connected_edge_index(x.device)
                xt = self.conv2(xt, edge_index)
            else:
                xt_refine_domain = self.target_adj_fc(x.T).T
                adj_t_refine = xt_refine_domain @ xt_refine_domain.T
                adj_t_refine = torch.sigmoid(adj_t_refine)
                adj_mask = (adj_t_refine > 0.5).float()
                adj_masked = adj_t_refine * adj_mask
                edge_index, edge_attr = dense_to_sparse(adj_masked)
                xt = self.conv2(xt, edge_index, edge_attr.view(-1, 1))

            xt = self.bn2(xt)
            xt = F.relu(xt)

        xt = xt @ xt.T  # (n_t, n_t)
        return xt

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)

        xt = self.linear_sr(x)

        if self.min_max_scale:
            xt_min = torch.min(xt)
            xt_max = torch.max(xt)
            xt = (xt - xt_min) / (xt_max - xt_min + 1e-8)

        if self.binarize:
            xt = (xt - torch.mean(xt)) / torch.std(xt)
            xt = torch.sigmoid(xt)

        return xt


class BiMP(nn.Module):
    """Bipartite Message Passing SR."""

    def __init__(
        self,
        n_source_nodes,
        n_target_nodes,
        target_node_embeddings,
        num_heads=4,
        edge_dim=1,
        dropout=0.2,
        beta=False,
        hidden_dim=32,
        refine_target=False,
        target_refine_fully_connected=True,
        min_max_scale=True,
        binarize=False,
    ):
        super().__init__()
        assert target_node_embeddings is not None
        self.n_target_nodes = n_target_nodes
        self.n_source_nodes = n_source_nodes
        self.min_max_scale = min_max_scale
        self.binarize = binarize

        self.conv1 = TransformerConv(
            n_source_nodes,
            hidden_dim,
            heads=num_heads,
            edge_dim=edge_dim,
            dropout=dropout,
            beta=beta,
        )
        self.bn1 = GraphNorm(hidden_dim * num_heads)

        self.target_node_embeddings = target_node_embeddings

        self.bipartite_attn = TransformerConv(
            hidden_dim * num_heads,
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            beta=beta,
        )
        self.bipartite_attn_bn = GraphNorm(hidden_dim * num_heads)

        self.refine_target = refine_target
        self.target_refine_fully_connected = target_refine_fully_connected
        if refine_target:
            if not target_refine_fully_connected:
                self.conv3 = TransformerConv(
                    hidden_dim * num_heads,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    beta=beta,
                )
                self.bn3 = GraphNorm(hidden_dim * num_heads)
                edge_dim = 1
            else:
                edge_dim = None

            self.conv2 = TransformerConv(
                hidden_dim * num_heads,
                hidden_dim,
                heads=num_heads,
                edge_dim=edge_dim,
                dropout=dropout,
                beta=beta,
            )
            self.bn2 = GraphNorm(hidden_dim * num_heads)

    def _fully_connected_edge_index(self, device):
        n = self.n_target_nodes
        row = torch.arange(n).repeat(n)
        col = torch.arange(n).repeat_interleave(n)
        return torch.stack([row, col], dim=0).to(device)

    def _bipartite_edge_index(self, device):
        n_s, n_t = self.n_source_nodes, self.n_target_nodes
        group1 = torch.arange(n_s)
        group2 = torch.arange(n_s, n_s + n_t)
        row = group1.repeat_interleave(n_t)
        col = group2.repeat(n_s)
        return torch.stack([row, col], dim=0).to(device)

    def attn_sr(self, x):
        x_bipartite = torch.cat([x, self.target_node_embeddings], dim=0)
        edge_index_bipartite = self._bipartite_edge_index(x.device)

        xt = self.bipartite_attn(x_bipartite, edge_index_bipartite)
        xt = self.bipartite_attn_bn(xt)
        xt = F.relu(xt)
        xt = xt[self.n_source_nodes:]  # (n_t, h)

        if self.refine_target:
            if self.target_refine_fully_connected:
                edge_index = self._fully_connected_edge_index(x.device)
                xt = self.conv2(xt, edge_index)
            else:
                x_bipartite_adj = torch.cat(
                    [x, self.target_node_embeddings], dim=0
                )
                xt_refine_domain = self.conv3(x_bipartite_adj, edge_index_bipartite)
                xt_refine_domain = self.bn3(xt_refine_domain)
                xt_refine_domain = xt_refine_domain[self.n_source_nodes:]

                adj_t_refine = xt_refine_domain @ xt_refine_domain.T
                adj_t_refine = torch.sigmoid(adj_t_refine)
                adj_mask = (adj_t_refine > 0.5).float()
                adj_masked = adj_t_refine * adj_mask
                edge_index, edge_attr = dense_to_sparse(adj_masked)
                xt = self.conv2(xt, edge_index, edge_attr.view(-1, 1))

            xt = self.bn2(xt)
            xt = F.relu(xt)

        xt = xt @ xt.T  # (n_t, n_t)
        return xt

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)

        xt = self.attn_sr(x)

        if self.min_max_scale:
            xt_min = torch.min(xt)
            xt_max = torch.max(xt)
            xt = (xt - xt_min) / (xt_max - xt_min + 1e-8)

        if self.binarize:
            xt = (xt - torch.mean(xt)) / torch.std(xt)
            xt = torch.sigmoid(xt)

        return xt


# ---------------------------------------------------------------------------
# Dual-domain wrapper
# ---------------------------------------------------------------------------


class DualLearner(nn.Module):
    def __init__(
        self, in_dim, out_dim=1, num_heads=4, dropout=0.2, beta=False,
        min_max_scale=True, binarize=False,
    ):
        super().__init__()
        if out_dim == 1:
            num_heads = 1
        else:
            assert out_dim % num_heads == 0

        self.min_max_scale = min_max_scale
        self.binarize = binarize

        self.conv1 = TransformerConv(
            in_dim, out_dim // num_heads, heads=num_heads, dropout=dropout, beta=beta
        )
        self.bn1 = GraphNorm(out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        xt = F.relu(x)

        if self.min_max_scale:
            xt_min = torch.min(xt)
            xt_max = torch.max(xt)
            xt = (xt - xt_min) / (xt_max - xt_min + 1e-8)

        if self.binarize:
            xt = (xt - torch.mean(xt)) / torch.std(xt)
            xt = torch.sigmoid(xt)

        return xt


class DualModel(nn.Module):
    def __init__(
        self, inner_model, n_target_nodes, dual_node_in_dim=1,
        dual_node_out_dim=1, num_heads=4, dropout=0.2, beta=False,
        min_max_scale=True, binarize=False,
    ):
        super().__init__()
        self.n_target_nodes = n_target_nodes
        self.n_ut = n_target_nodes * (n_target_nodes - 1) // 2

        self.node_init_model = inner_model

        self.dual_learner = DualLearner(
            dual_node_in_dim, dual_node_out_dim, num_heads, dropout, beta,
            min_max_scale, binarize,
        )

    def get_dual_node_init(self, source_data):
        dual_node_init_x = self.node_init_model(source_data)
        ut_mask = (
            torch.triu(
                torch.ones(self.n_target_nodes, self.n_target_nodes), diagonal=1
            )
            .bool()
            .to(dual_node_init_x.device)
        )
        dual_x = torch.masked_select(dual_node_init_x, ut_mask)
        dual_x = dual_x.view(self.n_ut, -1)
        return dual_x

    def forward(self, source_data, dual_edge_index):
        dual_node_init = self.get_dual_node_init(source_data)
        dual_target_x = self.dual_learner(dual_node_init, dual_edge_index)
        return dual_target_x


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------


def build_defend_model(cfg, device):
    """Build DEFEND model from Hydra config."""
    target_node_embeddings = None

    if cfg.model.sr_method == "bi_mp":
        target_node_embeddings = torch.randn(
            cfg.data.hr_dim,
            cfg.model.hidden_dim * cfg.model.num_heads,
            device=device,
        )

    if cfg.model.sr_method == "linear_algebraic":
        inner_model = LA(
            cfg.data.lr_dim,
            cfg.data.hr_dim,
            num_heads=cfg.model.num_heads,
            edge_dim=cfg.model.edge_dim,
            dropout=cfg.model.dropout,
            beta=cfg.model.beta,
            min_max_scale=cfg.model.min_max_scale,
            binarize=False,
        )
    elif cfg.model.sr_method == "bi_lc":
        inner_model = BiLC(
            cfg.data.lr_dim,
            cfg.data.hr_dim,
            num_heads=cfg.model.num_heads,
            edge_dim=cfg.model.edge_dim,
            dropout=cfg.model.dropout,
            beta=cfg.model.beta,
            hidden_dim=cfg.model.hidden_dim,
            refine_target=cfg.model.refine_target,
            target_refine_fully_connected=cfg.model.target_refine_fully_connected,
            min_max_scale=cfg.model.min_max_scale,
            binarize=False,
        )
    elif cfg.model.sr_method == "bi_mp":
        inner_model = BiMP(
            cfg.data.lr_dim,
            cfg.data.hr_dim,
            target_node_embeddings,
            num_heads=cfg.model.num_heads,
            edge_dim=cfg.model.edge_dim,
            dropout=cfg.model.dropout,
            beta=cfg.model.beta,
            hidden_dim=cfg.model.hidden_dim,
            refine_target=cfg.model.refine_target,
            target_refine_fully_connected=cfg.model.target_refine_fully_connected,
            min_max_scale=cfg.model.min_max_scale,
            binarize=False,
        )
    else:
        raise ValueError(f"Unknown sr_method: {cfg.model.sr_method}")

    if cfg.model.use_dual:
        dual_node_in_dim = 1
        if cfg.model.sr_method == "linear_algebraic":
            dual_node_in_dim = cfg.data.lr_dim
        elif cfg.model.sr_method in ("bi_lc", "bi_mp"):
            dual_node_in_dim = cfg.model.hidden_dim * cfg.model.num_heads

        model = DualModel(
            inner_model,
            cfg.data.hr_dim,
            dual_node_in_dim=dual_node_in_dim,
            dual_node_out_dim=cfg.model.dual_node_out_dim,
            num_heads=cfg.model.num_heads,
            dropout=cfg.model.dropout,
            beta=cfg.model.beta,
            min_max_scale=cfg.model.min_max_scale,
            binarize=cfg.model.binarize,
        )
    else:
        model = inner_model

    return model.to(device)
