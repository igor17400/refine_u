import torch
import torch.nn as nn


class GraphUnpool(nn.Module):
    def __init__(self):
        super(GraphUnpool, self).__init__()

    def forward(self, A, X, idx):
        new_X = torch.zeros([A.shape[0], X.shape[1]], device=X.device)
        new_X[idx] = X
        return A, new_X


class GraphPool(nn.Module):
    def __init__(self, k, in_dim):
        super(GraphPool, self).__init__()
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


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, A: torch.Tensor, X: torch.Tensor):
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
        super(SkipGate, self).__init__()
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, decoder_feat, encoder_feat):
        combined = torch.cat([decoder_feat, encoder_feat], dim=-1)
        g = torch.sigmoid(self.gate(combined))
        return decoder_feat + g * encoder_feat


class SymGraphUnet(nn.Module):
    def __init__(self, ks, in_dim, out_dim, hidden_dim, dropout, gated_skip=False):
        super(SymGraphUnet, self).__init__()
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
