import numpy as np
import torch
import torch.nn as nn
import random
from torch.autograd import Variable


class graphSAGELayer(nn.Module):
    def __init__(self, args, features_dim, out_feat_dim, device):
        super(graphSAGELayer, self).__init__()
        self.args = args
        self.features_dim = features_dim
        self.out_feat_dim = out_feat_dim
        self.device = device
        self.W = nn.Parameter(torch.empty(size=(self.features_dim * 2, self.out_feat_dim)))
        nn.init.xavier_uniform_(self.W.data)
        self.leakyrelu = nn.LeakyReLU(self.args.alpha)
        self.lstm = torch.nn.LSTM(input_size=self.features_dim, hidden_size=self.features_dim, bias=True, dropout=0.3,
                                  batch_first=True)

    def forward(self, h, adj_list, aggregate_num, aggregate_neighbors):
        h = h.to(self.device)
        # aggregate_neighbors = self.aggregate_neighbors_sample(adj_list, aggregate_num)
        if self.args.stru_aggregate_type == 'mean':
            neigh_feats = self.aggregate_neighbors_feats_func_mean(h, aggregate_neighbors)
        elif self.args.stru_aggregate_type == 'gcn':
            neigh_feats = self.aggregate_neighbors_feats_func_gcn(h, aggregate_neighbors)
        elif self.args.stru_aggregate_type == 'maxpool':
            neigh_feats = self.aggregate_neighbors_feats_func_maxpool(h, aggregate_neighbors)
        elif self.args.stru_aggregate_type == 'lstm':
            neigh_feats = self.aggregate_neighbors_feats_func_lstm(h, aggregate_neighbors)
        return neigh_feats

    def aggregate_neighbors_sample(self, adj_list, aggregate_num):
        # random.seed(self.args.seed)
        # np.random.seed(self.args.seed)
        adj_list = [list(set(neighbors)) for neighbors in adj_list]
        # adj_list = [neighbors for neighbors in adj_list]
        aggregate_neighbors = [np.random.choice(to_neigh, aggregate_num, replace=False) if len(
            to_neigh) >= aggregate_num else np.random.choice(to_neigh, aggregate_num, replace=True) for to_neigh in
                               adj_list]
        # aggregate_neighbors = [random.sample(to_neigh, aggregate_num) if len(to_neigh) >= aggregate_num
        #                        else np.random.choice(to_neigh, aggregate_num, replace=True) for to_neigh in adj_list]
        return aggregate_neighbors

    def aggregate_neighbors_feats_func_mean(self, h, aggregate_neighbors):
        mask = self.A_creat(aggregate_neighbors)
        feat = torch.mm(mask.to(self.device), h)
        num_neigh = mask.sum(1, keepdim=True).to(self.device)
        feat = feat.div(num_neigh)
        feat = torch.cat([h, feat], dim=1)
        return torch.mm(feat, self.W)

    def aggregate_neighbors_feats_func_gcn(self, h, aggregate_neighbors):
        mask = self.A_creat(aggregate_neighbors)
        mask = mask + torch.eye(mask.size(0))
        feat = torch.mm(mask.to(self.device), h)
        num_neigh = mask.sum(1, keepdim=True).to(self.device)
        feat = feat.div(num_neigh)
        feat = torch.cat([h, feat], dim=1)
        return torch.mm(feat, self.W)

    def aggregate_neighbors_feats_func_maxpool(self, h, aggregate_neighbors):
        result_tensor = torch.zeros(self.args.N, len(aggregate_neighbors[0]), h.size(1))
        for i in range(self.args.N):
            result_tensor[i] = h[aggregate_neighbors[i]]
        neighbors_feats = torch.max(result_tensor.to(self.device), dim=1)[0]
        neighbors_feats = torch.cat([h, neighbors_feats], dim=1)
        return torch.mm(neighbors_feats, self.W)

    def aggregate_neighbors_feats_func_lstm(self, h, aggregate_neighbors):
        result_tensor = torch.zeros(self.args.N, len(aggregate_neighbors[0]), h.size(1)).to(self.device)
        for i in range(self.args.N):
            result_tensor[i] = h[aggregate_neighbors[i]]
        h0 = torch.zeros(1, result_tensor.size(0), h.size(1)).to(self.device)
        c0 = torch.zeros(1, result_tensor.size(0), h.size(1)).to(self.device)
        output, _ = self.lstm(result_tensor, (h0, c0))
        neighbors_feats = output[:, -1, :]
        neighbors_feats = torch.cat([h, neighbors_feats], dim=1)
        return torch.mm(neighbors_feats, self.W)

    def A_creat(self, adj):
        nodes = list(range(len(adj)))
        mask = Variable(torch.zeros(len(adj), len(adj)))
        column_indices = [n for neigh in nodes for n in adj[neigh]]
        row_indices = [i for i in range(len(adj)) for _ in range(len(adj[i]))]
        for j in range(len(column_indices)):
            mask[row_indices[j], column_indices[j]] += 1
        return mask
