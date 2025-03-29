import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import graphSAGELayer
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
        self.sage_1_order = graphSAGELayer(self.args, self.graph, self.features_dim, self.args.hidden_dim,
                                           device=device).to(device)
        self.sage_2_order = graphSAGELayer(self.args, self.graph, self.args.hidden_dim, self.args.out_dim,
                                           device=device).to(device)
        self.W = nn.ParameterDict({})
        self.W_his = nn.Parameter(torch.empty(size=(self.args.out_dim, self.args.out_dim)))
        nn.init.xavier_uniform_(self.W_his.data)
        self.W_T = nn.Parameter(torch.empty(size=(self.args.out_dim * 2, self.args.out_dim)))
        nn.init.xavier_uniform_(self.W_T.data)
        self.leakyrelu = nn.LeakyReLU(self.args.alpha)
        self.lstm = torch.nn.LSTM(input_size=self.args.out_dim, hidden_size=self.args.out_dim, bias=True, dropout=0.2,
                                  batch_first=True).to(self.device)
        self.feats = nn.Parameter(torch.ones(args.N, args.first_dim).to(device), requires_grad=True)

        self._init_layers()

    def forward(self, graphs, agg_neigh_list1, agg_neigh_list2):
        graph = graphs[self.args.now]
        if self.args.now >= self.args.past:
            num = self.args.number[self.args.now - self.args.past]  # 统计具有至少past个历史快照的节点数
            user_num = self.args.user_number[self.args.now - self.args.past]
            item_num = self.args.item_number[self.args.now - self.args.past]
        feat_hidden1 = self.sage_1_order.forward(self.feats, agg_neigh_list1)  # 隐藏层输出
        feat_hidden1 = torch.dropout(feat_hidden1, 0.1, train=self.training)
        # feat_hidden1 = self.leakyrelu(feat_hidden1)
        feat_hidden1 = F.normalize(feat_hidden1, p=2., dim=1)
        feat_hidden1_copy = feat_hidden1
        if self.args.hidden_info and self.args.now >= self.args.past:
            # 使用隐藏层信息
            history_hidden_feature = []
            for i in range(self.args.past):
                history_hidden_feature.append(graphs[self.args.now - (i + 1)].hidden1)
            if self.args.time_aggregate_type == 'mean':
                user_features1, item_features1 = self.aggregate_neighbors_time_feats_func_mean(num, user_num, item_num,
                                                                                               feat_hidden1,
                                                                                               history_hidden_feature)
            elif self.args.time_aggregate_type == 'gcn':
                user_features1, item_features1 = self.aggregate_neighbors_time_feats_func_gcn(num, user_num, item_num,
                                                                                              feat_hidden1,
                                                                                              history_hidden_feature)
            elif self.args.time_aggregate_type == 'maxpool':
                user_features1, item_features1 = self.aggregate_neighbors_time_feats_func_maxpool(num, user_num,
                                                                                                  item_num,
                                                                                                  feat_hidden1,
                                                                                                  history_hidden_feature)
            elif self.args.time_aggregate_type == 'lstm':
                user_features1, item_features1 = self.aggregate_neighbors_time_feats_func_lstm(num, user_num, item_num,
                                                                                               feat_hidden1,
                                                                                               history_hidden_feature)
            user_features1 = torch.dropout(user_features1, 0.1, train=self.training)
            item_features1 = torch.dropout(item_features1, 0.1, train=self.training)
            user_features1 = self.leakyrelu(user_features1)
            item_features1 = self.leakyrelu(item_features1)
            user_features1 = F.normalize(user_features1, p=2., dim=1)
            item_features1 = F.normalize(item_features1, p=2., dim=1)
            feat_hidden1_copy = torch.cat((user_features1, feat_hidden1[user_num:self.args.user_N]), dim=0)
            feat_hidden1_copy = torch.cat((feat_hidden1_copy, item_features1), dim=0)
            feat_hidden1_copy = torch.cat((feat_hidden1_copy, feat_hidden1[self.args.user_N + item_num:]), dim=0)
        feat_hidden2 = self.sage_2_order.forward(feat_hidden1_copy, agg_neigh_list2)
        feat_hidden2 = torch.dropout(feat_hidden2, 0.1, train=self.training)
        # feat_hidden2 = self.leakyrelu(feat_hidden2)
        feat_hidden2 = F.normalize(feat_hidden2, p=2., dim=1)
        if self.args.now < self.args.past:
            return feat_hidden1, feat_hidden2, 'null'
        # 对时间维度邻居的操作
        history_feature = []
        for i in range(self.args.past):
            history_feature.append(graphs[self.args.now - (i + 1)].hidden2)
        if self.args.time_aggregate_type == 'mean':
            user_features, item_features = self.aggregate_neighbors_time_feats_func_mean(num, user_num, item_num,
                                                                                         feat_hidden2,
                                                                                         history_feature)
        elif self.args.time_aggregate_type == 'gcn':
            user_features, item_features = self.aggregate_neighbors_time_feats_func_gcn(num, user_num, item_num,
                                                                                        feat_hidden2,
                                                                                        history_feature)
        elif self.args.time_aggregate_type == 'maxpool':
            user_features, item_features = self.aggregate_neighbors_time_feats_func_maxpool(num, user_num, item_num,
                                                                                            feat_hidden2,
                                                                                            history_feature)
        elif self.args.time_aggregate_type == 'lstm':
            user_features, item_features = self.aggregate_neighbors_time_feats_func_lstm(num, user_num, item_num,
                                                                                         feat_hidden2,
                                                                                         history_feature)
        user_features = torch.dropout(user_features, 0.1, train=self.training)
        item_features = torch.dropout(item_features, 0.1, train=self.training)
        user_features = self.leakyrelu(user_features)
        item_features = self.leakyrelu(item_features)
        user_features = F.normalize(user_features, p=2., dim=1)
        item_features = F.normalize(item_features, p=2., dim=1)
        feat = torch.cat((user_features, feat_hidden2[user_num:self.args.user_N]), dim=0)
        feat = torch.cat((feat, item_features), dim=0)
        feat = torch.cat((feat, feat_hidden2[self.args.user_N + item_num:]), dim=0)
        return feat_hidden1, feat_hidden2, feat

    def aggregate_neighbors_time_feats_func_mean(self, num, user_num, item_num, feat, history_feature):
        time_feat = (torch.zeros(size=(num, self.args.out_dim))).to(self.device)
        for j in range(self.args.past):
            time_feat = torch.add(time_feat, torch.cat(
                [torch.mm(history_feature[j][:user_num], self.W_his.to(torch.double)),
                 torch.mm(history_feature[j][self.args.user_number[self.args.now - 1 - j]:
                                             self.args.user_number[self.args.now - 1 - j] + item_num],
                          self.W_his.to(torch.double))],
                dim=0))
        time_feat = torch.div(time_feat, self.args.past)
        return torch.mm(torch.cat([feat[:user_num], time_feat[:user_num]], dim=1), self.W_T.to(torch.double)), torch.mm(
            torch.cat([feat[self.args.user_N:self.args.user_N + item_num], time_feat[user_num:]], dim=1),
            self.W_T.to(torch.double))

    def aggregate_neighbors_time_feats_func_gcn(self, num, user_num, item_num, feat, history_feature):
        time_feat = torch.mm(feat[:num], self.W_his.double())
        for j in range(self.args.past):
            time_feat = torch.add(time_feat, torch.cat([torch.mm(history_feature[j][:user_num], self.W_his.double()),
                                                        torch.mm(history_feature[j][
                                                                 self.args.user_number[self.args.now - 1 - j]:
                                                                 self.args.user_number[self.args.now - 1 - j]
                                                                 + item_num],
                                                                 self.W_his.double())], dim=0))
        time_feat = torch.div(time_feat, self.args.past)
        return torch.mm(torch.cat([feat[:user_num], time_feat[:user_num]], dim=1), self.W_T.double()), torch.mm(
            torch.cat([feat[self.args.user_N:self.args.user_N + item_num], time_feat[user_num:]], dim=1),
            self.W_T.double())

    def aggregate_neighbors_time_feats_func_maxpool(self, num, user_num, item_num, feat, history_feature):
        time_feat = (torch.zeros(size=(self.args.past, num, feat.size(1)))).to(self.device)
        for i in range(self.args.past):
            time_feat[i] = torch.cat([torch.mm(history_feature[i][:user_num], self.W_his.double()),
                                      torch.mm(history_feature[i][
                                               self.args.user_number[self.args.now - 1 - i]:
                                               self.args.user_number[self.args.now - 1 - i]
                                               + item_num],
                                               self.W_his.double())], dim=0)
        time_feat = time_feat.permute(1, 0, 2)
        time_feat = torch.max(time_feat, dim=1)[0]
        return torch.mm(torch.cat([feat[:user_num], time_feat[:user_num]], dim=1), self.W_T.double()), torch.mm(
            torch.cat([feat[self.args.user_N:self.args.user_N + item_num], time_feat[user_num:]], dim=1),
            self.W_T.double())

    def aggregate_neighbors_time_feats_func_lstm(self, num, user_num, item_num, feat, history_feature):
        time_feat = (torch.zeros(size=(self.args.past, num, feat.size(1)))).to(self.device)
        for i in range(self.args.past):
            time_feat[i] = torch.cat([torch.mm(history_feature[i][:user_num], self.W_his.double()),
                                      torch.mm(history_feature[i][
                                               self.args.user_number[self.args.now - 1 - i]:
                                               self.args.user_number[self.args.now - 1 - i]
                                               + item_num],
                                               self.W_his.double())], dim=0)
        time_feat = time_feat.permute(1, 0, 2)
        h0 = torch.zeros(1, time_feat.size(0), feat.size(1)).to(self.device)
        c0 = torch.zeros(1, time_feat.size(0), feat.size(1)).to(self.device)
        output, _ = self.lstm(time_feat, (h0, c0))
        time_feat = output[:, -1, :]
        return torch.mm(torch.cat([feat[:user_num], time_feat[:user_num]], dim=1), self.W_T.double()), torch.mm(
            torch.cat([feat[self.args.user_N:self.args.user_N + item_num], time_feat[user_num:]], dim=1),
            self.W_T.double())

    def _init_layers(self):
        glorot(self.feats)
