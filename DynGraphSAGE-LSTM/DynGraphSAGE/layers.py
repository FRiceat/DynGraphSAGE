import torch
import torch.nn as nn
import torch.nn.functional as F


class graphSAGELayer(nn.Module):
    def __init__(self, args, features_dim, out_feat_dim, device):
        super(graphSAGELayer, self).__init__()
        self.args = args
        self.features_dim = features_dim
        self.out_feat_dim = out_feat_dim
        self.device = device
        self.W = nn.Parameter(torch.empty(size=(self.out_feat_dim * 2, self.out_feat_dim)))
        nn.init.xavier_uniform_(self.W.data)
        self.W_past = nn.Parameter(torch.empty(size=(self.features_dim, self.out_feat_dim)))
        nn.init.xavier_uniform_(self.W_past.data)
        self.leakyrelu = nn.LeakyReLU(self.args.alpha)
        self.lstm = torch.nn.LSTM(input_size=self.out_feat_dim, hidden_size=self.out_feat_dim, bias=True, dropout=0.3,
                                  batch_first=True)

    def forward(self, h, adj_list, aggregate_num, graphs, aggregate_neighbors):
        h = h.to(self.device)
        neigh_feats, h_n, c_n = self.dyngraphsage_lstm(h, aggregate_neighbors, graphs)
        return neigh_feats, h_n, c_n

    def dyngraphsage_lstm(self, h, aggregate_neighbors, graphs):
        # lstm_layer = self.lstm(input_size=self.features_dim, hidden_size=self.features_dim, bias=True, dropout=0.1,
        #                        batch_first=True).to(self.device)
        result_tensor = torch.zeros(self.args.N, len(aggregate_neighbors[0]), self.out_feat_dim).to(self.device)
        h = torch.mm(h, self.W_past)
        for i in range(self.args.N):
            result_tensor[i] = h[aggregate_neighbors[i]]
        if self.args.now > 0:
            graph = graphs[self.args.now - 1]
            node_number_now = self.args.number[self.args.now]
            node_number_past = self.args.number[self.args.now - 1]
            jiange = node_number_now - node_number_past
            if self.args.biaoshi == 0:
                h0 = torch.cat([graph.h_t1, torch.zeros(jiange, self.out_feat_dim).to(self.device)], dim=0).unsqueeze(0)
                c0 = torch.cat([graph.c_t1, torch.zeros(jiange, self.out_feat_dim).to(self.device)], dim=0).unsqueeze(0)
            else:
                h0 = torch.cat([graph.h_t2, torch.zeros(jiange, self.out_feat_dim).to(self.device)], dim=0).unsqueeze(0)
                c0 = torch.cat([graph.c_t2, torch.zeros(jiange, self.out_feat_dim).to(self.device)], dim=0).unsqueeze(0)

        else:
            h0 = torch.zeros(1, result_tensor.size(0), self.out_feat_dim).to(self.device)
            c0 = torch.zeros(1, result_tensor.size(0), self.out_feat_dim).to(self.device)
        output, (h_n, c_n) = self.lstm(result_tensor, (h0, c0))
        h_n = h_n.squeeze(0)
        c_n = c_n.squeeze(0)
        neighbors_feats = h_n
        neighbors_feats = torch.cat([h, neighbors_feats], dim=1)
        return F.elu(torch.matmul(neighbors_feats, self.W)), h_n, c_n
