import numpy as np
import torch
import torch.nn as nn
import random
from torch.autograd import Variable
import torch.nn.functional as F


class graphSAGELayer(nn.Module):
    def __init__(self, args, graph, features_dim, out_feat_dim, device):
        super(graphSAGELayer, self).__init__()
        self.args = args
        self.graph = graph
        self.features_dim = features_dim
        self.out_feat_dim = out_feat_dim
        self.device = device
        self.W = nn.Parameter(
            torch.empty(size=(self.features_dim * 2 + self.graph.graph['edge_feature'][1].shape[0], self.out_feat_dim)))
        nn.init.xavier_uniform_(self.W.data)
        self.leakyrelu = nn.LeakyReLU(self.args.alpha)
        self.lstm = torch.nn.LSTM(input_size=self.features_dim, hidden_size=self.features_dim, bias=True, dropout=0.1,
                                  batch_first=True)

    def forward(self, h, aggregate_neighbors):
        # h = h.to(self.device)
        # aggregate_neighbors = self.aggregate_neighbors_sample(adj_list, aggregate_num)
        if self.args.stru_aggregate_type == 'mean':
            neigh_feats = self.aggregate_neighbors_feats_func_mean(h, aggregate_neighbors)
        elif self.args.stru_aggregate_type == 'gcn':
            neigh_feats = self.aggregate_neighbors_feats_func_gcn(h, aggregate_neighbors)
        elif self.args.stru_aggregate_type == 'maxpool':
            neigh_feats = self.aggregate_neighbors_feats_func_maxpool(h, aggregate_neighbors)
        elif self.args.stru_aggregate_type == 'lstm':
            neigh_feats = self.aggregate_neighbors_feats_func_lstm(h, aggregate_neighbors)
        return F.elu(neigh_feats)

    def aggregate_neighbors_feats_func_mean(self, h, aggregate_neighbors):
        edge_features = self.graph.graph['edge_feature']
        all_neighbor_indices = []
        all_edge_indices = []
        for i in range(len(aggregate_neighbors)):
            if aggregate_neighbors[i] == [-1]:
                all_neighbor_indices.append(i)
                all_edge_indices.append(-1)
                continue
            for neighbor, edge_id in aggregate_neighbors[i]:
                all_neighbor_indices.append(neighbor)
                all_edge_indices.append(edge_id)
        all_neighbor_indices = torch.tensor(all_neighbor_indices, dtype=torch.int64)
        all_edge_indices = torch.tensor(all_edge_indices)
        neighbor_features = h[all_neighbor_indices]
        edge_features_batch = [edge_features[int(idx)] for idx in all_edge_indices]
        edge_features_batch = torch.cat([tensor.view(-1, tensor.size(-1)) for tensor in edge_features_batch],
                                        dim=0)
        concatenated_features = torch.cat([neighbor_features, edge_features_batch.to(self.device)], dim=1)
        grouped_indices = torch.cumsum(torch.tensor([0] + [len(neighbors) for neighbors in aggregate_neighbors]), dim=0)
        means = []
        for start, end in zip(grouped_indices, grouped_indices[1:]):
            if start < end:
                means.append(concatenated_features[start:end].mean(dim=0))

        aggregate_features = torch.stack(means)
        feat = torch.cat([h, aggregate_features], dim=1)
        return torch.mm(feat, self.W.to(torch.double))

    def aggregate_neighbors_feats_func_gcn(self, h, aggregate_neighbors):
        edge_features = self.graph.graph['edge_feature']
        all_neighbor_indices = []
        all_edge_indices = []
        for i in range(len(aggregate_neighbors)):
            all_neighbor_indices.append(i)
            all_edge_indices.append(-1)
            if aggregate_neighbors[i] == [-1]:
                all_neighbor_indices.append(i)
                all_edge_indices.append(-1)
                continue
            for neighbor, edge_id in aggregate_neighbors[i]:
                all_neighbor_indices.append(neighbor)
                all_edge_indices.append(edge_id)
        all_neighbor_indices = torch.tensor(all_neighbor_indices, dtype=torch.int64)
        all_edge_indices = torch.tensor(all_edge_indices)
        neighbor_features = h[all_neighbor_indices]
        edge_features_batch = [edge_features[int(idx)] for idx in all_edge_indices]
        edge_features_batch = torch.cat([tensor.view(-1, tensor.size(-1)) for tensor in edge_features_batch],
                                        dim=0)
        concatenated_features = torch.cat([neighbor_features, edge_features_batch.to(self.device)], dim=1)
        grouped_indices = torch.cumsum(torch.tensor([0] + [len(neighbors) + 1 for neighbors in aggregate_neighbors]),
                                       dim=0)
        means = []
        for start, end in zip(grouped_indices, grouped_indices[1:]):
            if start < end:
                means.append(concatenated_features[start:end].mean(dim=0))

        aggregate_features = torch.stack(means)
        feat = torch.cat([h, aggregate_features], dim=1)
        return torch.mm(feat, self.W.to(torch.double))

    def aggregate_neighbors_feats_func_maxpool(self, h, aggregate_neighbors):
        edge_features = self.graph.graph['edge_feature']
        all_neighbor_indices = []
        all_edge_indices = []
        for i in range(len(aggregate_neighbors)):
            all_neighbor_indices.append(i)
            all_edge_indices.append(-1)
            if aggregate_neighbors[i] == [-1]:
                all_neighbor_indices.append(i)
                all_edge_indices.append(-1)
                continue
            for neighbor, edge_id in aggregate_neighbors[i]:
                all_neighbor_indices.append(neighbor)
                all_edge_indices.append(edge_id)
        all_neighbor_indices = torch.tensor(all_neighbor_indices, dtype=torch.int64)
        all_edge_indices = torch.tensor(all_edge_indices)
        neighbor_features = h[all_neighbor_indices]
        edge_features_batch = [edge_features[int(idx)] for idx in all_edge_indices]
        edge_features_batch = torch.cat([tensor.view(-1, tensor.size(-1)) for tensor in edge_features_batch],
                                        dim=0)
        concatenated_features = torch.cat([neighbor_features, edge_features_batch.to(self.device)], dim=1)
        grouped_indices = torch.cumsum(torch.tensor([0] + [len(neighbors) + 1 for neighbors in aggregate_neighbors]),
                                       dim=0)
        maxes = []
        for start, end in zip(grouped_indices, grouped_indices[1:]):
            if start < end:
                maxes.append(concatenated_features[start:end].max(dim=0)[0])
        aggregate_features = torch.stack(maxes)
        feat = torch.cat([h, aggregate_features], dim=1)
        return torch.mm(feat, self.W.to(torch.double))

    def aggregate_neighbors_feats_func_lstm(self, h, aggregate_neighbors):

        edge_features = self.graph.graph['edge_feature']
        all_neighbor_indices = []
        all_edge_indices = []
        for i in range(len(aggregate_neighbors)):
            if aggregate_neighbors[i] == [-1]:
                all_neighbor_indices.append(i)
                all_edge_indices.append(-1)
                continue
            for neighbor, edge_id in aggregate_neighbors[i]:
                all_neighbor_indices.append(neighbor)
                all_edge_indices.append(edge_id)
        all_neighbor_indices = torch.tensor(all_neighbor_indices, dtype=torch.int64)
        all_edge_indices = torch.tensor(all_edge_indices)
        neighbor_features = h[all_neighbor_indices]
        edge_features_batch = [edge_features[int(idx)] for idx in all_edge_indices]
        edge_features_batch = torch.cat([tensor.view(-1, tensor.size(-1)) for tensor in edge_features_batch],
                                        dim=0)
        concatenated_features = torch.cat([neighbor_features, edge_features_batch.to(self.device)], dim=1)
        grouped_indices = torch.cumsum(torch.tensor([0] + [len(neighbors) for neighbors in aggregate_neighbors]), dim=0)
        means = []
        for start, end in zip(grouped_indices, grouped_indices[1:]):
            if start < end:
                means.append(concatenated_features[start:end].mean(dim=0))

        aggregate_features = torch.stack(means)
        feat = torch.cat([h, aggregate_features], dim=1)

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
