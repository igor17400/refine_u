import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RefineU building blocks
# ---------------------------------------------------------------------------


class GraphPool(nn.Module):
    def __init__(self, k, in_dim):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, X):
        scores = self.proj(X)
        scores = torch.squeeze(scores)
        scores = self.sigmoid(scores / 100)
        num_nodes = A.shape[0]
        values, idx = torch.topk(scores, int(self.k * num_nodes))
        new_X = X[idx, :]
        values = torch.unsqueeze(values, -1)
        new_X = torch.mul(new_X, values)
        A = A[idx, :]
        A = A[:, idx]
        return A, new_X, idx


class GraphUnpool(nn.Module):
    def forward(self, A, X, idx):
        new_X = torch.zeros([A.shape[0], X.shape[1]], device=X.device)
        new_X[idx] = X
        return A, new_X


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, A, X):
        X = self.drop(X)

        A_hat = A + torch.eye(A.size(0), device=A.device)
        rowsum = A_hat.sum(1)
        r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.0
        r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
        A_hat = torch.matmul(A_hat, r_mat_inv_sqrt)
        A_hat = torch.transpose(A_hat, 0, 1)
        A_hat = torch.matmul(A_hat, r_mat_inv_sqrt)

        X = torch.matmul(A_hat, X)
        X = self.proj(X)
        return X


class SkipGate(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, decoder_feat, encoder_feat):
        combined = torch.cat([decoder_feat, encoder_feat], dim=-1)
        g = torch.sigmoid(self.gate(combined))
        return decoder_feat + g * encoder_feat


# ---------------------------------------------------------------------------
# RefineU: Symmetric Graph U-Net with TopK pooling
# ---------------------------------------------------------------------------


class RefineU(nn.Module):
    def __init__(self, ks, in_dim, out_dim, hidden_dim, dropout, gated_skip=False):
        super().__init__()
        self.ks = ks
        self.gated_skip = gated_skip

        self.start_gcn = GCN(in_dim, hidden_dim, dropout=dropout)
        self.bottom_gcn = GCN(hidden_dim, hidden_dim, dropout=dropout)
        self.end_gcn = GCN(hidden_dim, out_dim, dropout=dropout)
        self.down_gcns = nn.ModuleList([])
        self.up_gcns = nn.ModuleList([])
        self.pools = nn.ModuleList([])
        self.unpools = nn.ModuleList([])
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(GCN(hidden_dim, hidden_dim, dropout=dropout))
            self.up_gcns.append(GCN(hidden_dim, hidden_dim, dropout=dropout))
            self.pools.append(GraphPool(ks[i], hidden_dim))
            self.unpools.append(GraphUnpool())

        if gated_skip:
            self.skip_gates = nn.ModuleList(
                [SkipGate(hidden_dim) for _ in range(self.l_n)]
            )

    def forward(self, A, X):
        adj_ms = []
        indices_list = []
        down_outs = []
        X = self.start_gcn(A, X)
        start_gcn_outs = X
        for i in range(self.l_n):
            X = self.down_gcns[i](A, X)
            adj_ms.append(A)
            down_outs.append(X)
            A, X, idx = self.pools[i](A, X)
            indices_list.append(idx)
        X = self.bottom_gcn(A, X)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1

            A, idx = adj_ms[up_idx], indices_list[up_idx]
            A, X = self.unpools[i](A, X, idx)
            X = self.up_gcns[i](A, X)
            if self.gated_skip:
                X = self.skip_gates[i](X, down_outs[up_idx])
            else:
                X = X.add(down_outs[up_idx])
        end_gcn_outs = X
        X = self.end_gcn(A, X)

        return X, start_gcn_outs, end_gcn_outs


# ---------------------------------------------------------------------------
# DenseGCNLayer: batched GCN for the traditional (non-UNet) baseline
# ---------------------------------------------------------------------------


class DenseGCNLayer(nn.Module):
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


# ---------------------------------------------------------------------------
# GraphCentMapperModel: full model (RefineU or traditional GCN + MLP head)
# ---------------------------------------------------------------------------


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
        output_activation="none",
    ):
        super().__init__()
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.gcn_in_dim = gcn_in
        self.gcn_hidden_dim = gcn_hidden
        self.gcn_out_dim = gcn_out
        self.gcn_type = gcn_type
        self.low_rank_k = low_rank_k
        self.output_activation = output_activation

        if self.gcn_type == "unet":
            if pool_ratios is None:
                pool_ratios = [0.5, 0.5, 0.5, 0.5]
            self.unet = RefineU(
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

        if self.output_activation == "leaky_relu":
            out_adj = F.leaky_relu(out_adj)
        elif self.output_activation == "relu":
            out_adj = F.relu(out_adj)

        return out_adj, x_temp_in, x_temp_out
