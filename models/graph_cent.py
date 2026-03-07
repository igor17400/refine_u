import torch
import torch.nn as nn
import torch.nn.functional as F

from refine_u.models.ops import SymGraphUnet


class DenseGCNLayer(nn.Module):
    """A simple GCN layer that operates on dense adjacency matrices."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, adj, x):
        batch_size, num_nodes, _ = adj.shape

        I = (
            torch.eye(num_nodes, device=adj.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        A_hat = adj + I

        D_hat = torch.sum(A_hat, dim=-1)
        D_hat_inv_sqrt = torch.pow(D_hat + 1e-8, -0.5)
        D_hat_inv_sqrt_mat = torch.diag_embed(D_hat_inv_sqrt)

        norm_adj = torch.bmm(D_hat_inv_sqrt_mat, A_hat)
        norm_adj = torch.bmm(norm_adj, D_hat_inv_sqrt_mat)

        support = torch.matmul(x, self.weight)
        output = torch.bmm(norm_adj, support) + self.bias
        return output


class GraphCentMapperModel(nn.Module):
    def __init__(
        self,
        input_nodes=160,
        output_nodes=268,
        gcn_in=32,
        gcn_hidden=64,
        gcn_out=32,
        mlp_hidden=256,
        dropout=0.2,
        gcn_type="unet",
        pool_ratios=None,
        gated_skip=False,
        low_rank_k=0,
    ):
        super().__init__()
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.gcn_in_dim = gcn_in
        self.gcn_hidden_dim = gcn_hidden
        self.gcn_out_dim = gcn_out
        self.gcn_type = gcn_type
        self.low_rank_k = low_rank_k

        if self.gcn_type == "unet":
            if pool_ratios is None:
                pool_ratios = [0.5, 0.5, 0.5, 0.5]
            self.unet = SymGraphUnet(
                pool_ratios,
                self.gcn_in_dim,
                self.gcn_out_dim,
                self.gcn_hidden_dim,
                dropout,
                gated_skip=gated_skip,
            )
        elif self.gcn_type == "traditional":
            self.gcn1 = DenseGCNLayer(self.gcn_in_dim, self.gcn_hidden_dim)
            self.gcn2 = DenseGCNLayer(self.gcn_hidden_dim, self.gcn_out_dim)
        else:
            raise ValueError(f"Unsupported gcn_type: {self.gcn_type}")

        flattened_size = self.input_nodes * self.gcn_out_dim

        if low_rank_k > 0:
            target_size = self.output_nodes * low_rank_k
        else:
            target_size = self.output_nodes * self.output_nodes

        self.mlp = nn.Sequential(
            nn.Linear(flattened_size, mlp_hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(mlp_hidden, target_size),
        )

    def forward(self, adj, feat):
        batch_size = adj.size(0)

        x_temp_in = torch.ones(
            batch_size, self.input_nodes, self.gcn_hidden_dim, device=adj.device
        )
        x_temp_out = torch.ones(
            batch_size, self.input_nodes, self.gcn_hidden_dim, device=adj.device
        )
        x_outs = torch.ones(
            batch_size, self.input_nodes, self.gcn_out_dim, device=adj.device
        )

        if self.gcn_type == "unet":
            for i in range(batch_size):
                x_outs[i], x_temp_in[i], x_temp_out[i] = self.unet(adj[i], feat[i])
            h = x_outs

        elif self.gcn_type == "traditional":
            x = F.relu(self.gcn1(adj, feat))
            x = F.relu(self.gcn2(adj, x))
            h = x

        else:
            raise ValueError(f"Unsupported gcn_type: {self.gcn_type}")

        h = h.view(batch_size, -1)
        h = self.mlp(h)

        if self.low_rank_k > 0:
            U = h.view(batch_size, self.output_nodes, self.low_rank_k)
            out_adj = torch.bmm(U, U.transpose(1, 2))
        else:
            out_adj = h.view(batch_size, self.output_nodes, self.output_nodes)
            out_adj = (out_adj + out_adj.transpose(1, 2)) / 2.0

        return out_adj, x_temp_in, x_temp_out
