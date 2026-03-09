import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_variable_glorot(output_dim):
    input_dim = output_dim
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = np.random.uniform(-init_range, init_range, (input_dim, output_dim))
    return initial


def normalize_adj_torch(mx):
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = torch.transpose(mx, 0, 1)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    return mx


def pad_HR_adj(label, split):
    label = np.pad(label, ((split, split), (split, split)), mode="constant")
    np.fill_diagonal(label, 1)
    return torch.from_numpy(label).type(torch.FloatTensor)


def unpad(data, split):
    idx_0 = data.shape[0] - split
    idx_1 = data.shape[1] - split
    train = data[split:idx_0, split:idx_1]
    return train


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
    """GSRN's GCN: linear projection only (no adjacency normalization)."""

    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(p=0)

    def forward(self, A, X):
        X = self.drop(X)
        X = self.proj(X)
        return X


class GraphUnet(nn.Module):
    """GSRN's GraphUnet: uses plain lists (not nn.ModuleList), concatenation skip."""

    def __init__(self, ks, in_dim, out_dim, dim=320):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.start_gcn = GCN(in_dim, dim)
        self.bottom_gcn = GCN(dim, dim)
        self.end_gcn = GCN(2 * dim, out_dim)
        self.down_gcns = []
        self.up_gcns = []
        self.pools = []
        self.unpools = []
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim))
            self.up_gcns.append(GCN(dim, dim))
            self.pools.append(GraphPool(ks[i], dim))
            self.unpools.append(GraphUnpool())

    def forward(self, A, X):
        adj_ms = []
        indices_list = []
        down_outs = []
        X = self.start_gcn(A, X)
        start_gcn_outs = X
        org_X = X
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
            X = X.add(down_outs[up_idx])
        X = torch.cat([X, org_X], 1)
        X = self.end_gcn(A, X)
        return X, start_gcn_outs


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0, act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        return output


class GSRLayer(nn.Module):
    def __init__(self, hr_dim):
        super(GSRLayer, self).__init__()
        self.hr_dim = hr_dim
        self.weights = torch.from_numpy(weight_variable_glorot(hr_dim)).type(
            torch.FloatTensor
        )
        self.weights = nn.Parameter(data=self.weights, requires_grad=True)

    def forward(self, A, X):
        lr = A
        lr_dim = lr.shape[0]
        f = X
        eig_val_lr, U_lr = torch.linalg.eigh(lr, UPLO="U")
        eye_mat = torch.eye(lr_dim, device=A.device)
        s_d = torch.cat((eye_mat, eye_mat), 0)

        a = torch.matmul(self.weights, s_d)
        b = torch.matmul(a, torch.t(U_lr))
        f_d = torch.matmul(b, f)
        f_d = torch.abs(f_d)
        self.f_d = f_d.fill_diagonal_(1)
        adj = normalize_adj_torch(self.f_d)
        X = torch.mm(adj, adj.t())
        X = (X + X.t()) / 2
        idx = torch.eye(X.shape[0], dtype=bool, device=X.device)
        X[idx] = 1
        return adj, torch.abs(X)


class GSRNet(nn.Module):
    def __init__(self, ks, lr_dim, hr_dim, hidden_dim, padding):
        super(GSRNet, self).__init__()
        self.lr_dim = lr_dim
        self.hr_dim = hr_dim
        self.hidden_dim = hidden_dim
        self.padding = padding
        self.padded_dim = hr_dim + 2 * padding

        self.layer = GSRLayer(self.padded_dim)
        self.net = GraphUnet(ks, self.lr_dim, self.padded_dim, dim=self.padded_dim)
        self.gc1 = GraphConvolution(self.padded_dim, self.hidden_dim, 0, act=F.relu)
        self.gc2 = GraphConvolution(self.hidden_dim, self.padded_dim, 0, act=F.relu)

    def forward(self, lr):
        I = torch.eye(self.lr_dim, device=lr.device)
        A = normalize_adj_torch(lr)

        self.net_outs, self.start_gcn_outs = self.net(A, I)

        self.outputs, self.Z = self.layer(A, self.net_outs)

        self.hidden1 = self.gc1(self.Z, self.outputs)
        self.hidden2 = self.gc2(self.hidden1, self.outputs)

        z = self.hidden2
        z = (z + z.t()) / 2
        idx = torch.eye(z.shape[0], dtype=bool, device=z.device)
        z[idx] = 1

        return torch.abs(z), self.net_outs, self.start_gcn_outs, self.outputs
