import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import AttentionLayer
import warnings
from torch_geometric.nn.inits import glorot

warnings.simplefilter("ignore")


class DynGraphSAGE(nn.Module):
    def __init__(self, args, graph, features_dim, device):
        super(DynGraphSAGE, self).__init__()
        self.args = args
        self.graph = graph
        self.features_dim = features_dim
        self.device = device
        # self.sage_1_order = graphSAGELayer(self.args, self.features_dim, self.args.hidden_dim, device=device).to(device)
        # self.sage_2_order = graphSAGELayer(self.args, self.args.hidden_dim, self.args.out_dim, device=device).to(device)
        self.sage_1_order = AttentionLayer(self.args, self.features_dim, self.args.hidden_dim, device=device).to(device)
        self.sage_2_order = AttentionLayer(self.args, self.args.hidden_dim * args.heads, self.args.out_dim,
                                           device=device).to(device)
        self.W = nn.Parameter(torch.empty(size=(self.features_dim, self.args.out_dim))).to(device)
        nn.init.xavier_uniform_(self.W.data)
        self.W_his = nn.Parameter(torch.empty(size=(self.args.out_dim, self.args.out_dim))).to(device)
        nn.init.xavier_uniform_(self.W_his.data)
        self.W_T = nn.Parameter(torch.empty(size=(self.args.out_dim * 2, self.args.out_dim))).to(device)
        nn.init.xavier_uniform_(self.W_T.data)
        self.leakyrelu = nn.LeakyReLU(self.args.alpha).to(device)
        self.lstm = torch.nn.LSTM(input_size=self.args.out_dim, hidden_size=self.args.out_dim, bias=True, dropout=0.1,
                                  batch_first=True).to(self.device)
        self.feats = nn.Parameter(torch.ones(args.number[-1], args.first_dim).to(device))
        # nn.init.xavier_uniform_(self.feats)
        # self.feats = torch.FloatTensor(np.eye(self.args.number[-1])[:args.N]).to(self.device)
        self.W_head1 = nn.Parameter(torch.empty(size=(self.args.out_dim * self.args.heads, self.args.out_dim))).to(
            self.device)
        nn.init.xavier_uniform_(self.W_head1.data)
        self.W_head2 = nn.Parameter(torch.empty(size=(self.args.out_dim * self.args.heads, self.args.out_dim))).to(
            self.device)
        nn.init.xavier_uniform_(self.W_head2.data)

        self._init_layers()

    def forward(self, graphs, adj_list, agg_neigh_list1, agg_neigh_list2):
        graph = graphs[self.args.now]
        h = torch.dropout((torch.LongTensor(graph.graph['feature'].todense()) / 1.).to(self.device), 0.1,
                          train=self.training)
        for i in range(self.args.heads):
            # feat_hidden1 = self.sage_1_order.forward(self.feats[:self.args.N], adj_list)  # 隐藏层输出
            feat_hidden1 = self.sage_1_order.forward(h, adj_list)
            if i.__eq__(0):
                feat_head1 = feat_hidden1
            else:
                feat_head1 = torch.cat([feat_head1, feat_hidden1], dim=1)

        feat_hidden1 = torch.dropout(feat_head1, 0.1, train=self.training)
        feat_hidden1_copy = feat_hidden1

        feat_hidden2 = self.sage_2_order.forward(feat_hidden1_copy, adj_list)
        feat_hidden2 = F.normalize(feat_hidden2, p=2., dim=1)

        if self.args.now < self.args.past:
            return feat_hidden1, feat_hidden2, 'null'
        # 对时间维度邻居的操作
        num = self.args.number[self.args.now - self.args.past]  # 统计具有至少past个历史快照的节点数
        history_feature = []
        for i in range(self.args.past):
            history_feature.append(graphs[self.args.now - (i + 1)].hidden2)
        if self.args.time_aggregate_type == 'mean':
            features = self.aggregate_neighbors_time_feats_func_mean(num, feat_hidden2, history_feature)
        elif self.args.time_aggregate_type == 'gcn':
            features = self.aggregate_neighbors_time_feats_func_gcn(num, feat_hidden2, history_feature)
        elif self.args.time_aggregate_type == 'maxpool':
            features = self.aggregate_neighbors_time_feats_func_maxpool(num, feat_hidden2, history_feature)
        elif self.args.time_aggregate_type == 'lstm':
            features = self.aggregate_neighbors_time_feats_func_lstm(num, feat_hidden2, history_feature)
        features = torch.dropout(features, 0.1, train=self.training)
        features = self.leakyrelu(features)
        features = F.normalize(features, p=2., dim=1)
        feat = torch.cat((features, feat_hidden2[num:]), dim=0)
        return feat_hidden1, feat_hidden2, feat

    def aggregate_neighbors_time_feats_func_mean(self, num, feat, history_feature):
        time_feat = (torch.zeros(size=(num, self.args.out_dim))).to(self.device)
        for j in range(self.args.past):
            time_feat = torch.add(time_feat, torch.mm(history_feature[j][:num], self.W_his))
        time_feat = torch.div(time_feat, self.args.past)
        return torch.mm(torch.cat([feat[:num], time_feat], dim=1), self.W_T)

    def aggregate_neighbors_time_feats_func_gcn(self, num, feat, history_feature):
        time_feat = torch.mm(feat[:num], self.W_his)
        for j in range(self.args.past):
            time_feat = torch.add(time_feat, torch.mm(history_feature[j][:num], self.W_his))
        time_feat = torch.div(time_feat, self.args.past + 1)
        return torch.mm(torch.cat([feat[:num], time_feat], dim=1), self.W_T)

    def aggregate_neighbors_time_feats_func_maxpool(self, num, feat, history_feature):
        time_feat = (torch.zeros(size=(self.args.past, num, feat.size(1)))).to(self.device)
        for i in range(self.args.past):
            time_feat[i] = torch.mm(history_feature[i][:num], self.W_his)
        time_feat = time_feat.permute(1, 0, 2)
        time_feat = torch.max(time_feat, dim=1)[0]
        return torch.mm(torch.cat([feat[:num], time_feat], dim=1), self.W_T)

    def aggregate_neighbors_time_feats_func_lstm(self, num, feat, history_feature):
        time_feat = (torch.zeros(size=(self.args.past, num, feat.size(1)))).to(self.device)
        for i in range(self.args.past):
            time_feat[i] = torch.mm(history_feature[i][:num], self.W_his)
        time_feat = time_feat.permute(1, 0, 2)
        h0 = torch.zeros(1, time_feat.size(0), feat.size(1)).to(self.device)
        c0 = torch.zeros(1, time_feat.size(0), feat.size(1)).to(self.device)
        output, _ = self.lstm(time_feat, (h0, c0))
        time_feat = output[:, -1, :]
        return torch.mm(torch.cat([feat[:num], time_feat], dim=1), self.W_T)

    def _init_layers(self):
        glorot(self.feats[:self.args.N])
