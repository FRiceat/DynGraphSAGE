import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, args, graph, adj_list):
        super(MyDataset, self).__init__()
        self.args = args
        self.graph = graph  # graph.edge[x][y]，输出是图中节点x和节点y交互的时间点
        self.adj_list = adj_list
        self.train_nodes = list(range(max(self.graph.nodes()) + 1))
        self.__createitems__()

    def __len__(self):
        return max(self.graph.nodes()) + 1

    def __getitem__(self, index):
        node = self.train_nodes[index]  # 涉及到到所有节点
        return self.data_items[node]  # 该节点对应到的所有信息

    def __createitems__(self):
        self.data_items = {}
        for i in range(self.args.N):
            feed_dict = {}
            node_1 = []
            node_pos_1 = []
            node_neg_1 = []
            node_1.append([i])
            pos_sample, neg_sample = self.sampling(i)
            node_pos_1.append(pos_sample)
            node_neg_1.append(neg_sample)
            node_1_list = [torch.LongTensor(node) for node in node_1]
            node_pos_list = [torch.LongTensor(node) for node in node_pos_1]
            node_neg_list = [torch.LongTensor(node) for node in node_neg_1]
            feed_dict['source'] = node_1_list  # 源节点
            feed_dict['positive'] = node_pos_list  # 源节点正采样节点
            feed_dict['negitive'] = node_neg_list  # 源节点负采样节点
            self.data_items[i] = feed_dict

    def sampling(self, node):
        neighbors_pos = list(set(self.adj_list[node]))
        neighbors_neg = list(set(range(self.args.N)) - set(neighbors_pos) - {node})
        if len(neighbors_pos) >= self.args.pos_num:
            pos_sample = list(np.random.choice(neighbors_pos, size=self.args.pos_num, replace=False))
        else:
            if len(self.adj_list[node]) == 0:
                self.adj_list[node].append(node)
            pos_sample = list(np.random.choice(self.adj_list[node], size=self.args.pos_num, replace=True))

        if len(neighbors_neg) >= self.args.neg_num:
            neg_sample = list(np.random.choice(neighbors_neg, size=self.args.neg_num, replace=False))
        else:
            neg_sample = list(np.random.choice(neighbors_neg, size=self.args.neg_num, replace=True))
        return pos_sample, neg_sample
