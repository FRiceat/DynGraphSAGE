import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
import shutil
import networkx as nx


def copy_data(args):
    shutil.copyfile("../data/%s/%s.pkl" % (args.data_name, args.data_name),
                    "../embedding/%s/%s.pkl" % (args.data_name, args.data_name))


def load_data(args):
    with open("../embedding/%s/%s.pkl" % (args.data_name, args.data_name), "rb") as f:
        graphs = pkl.load(f)  # 加载图
    adjs = [nx.adjacency_matrix(g) for g in graphs]
    args.snapshot_num = len(graphs)  # 统计图中时间快照的数量
    for i in range(args.snapshot_num):  # 统计各张时间快照中节点数量
        args.number.append(max(graphs[i].nodes) + 1)
    return graphs, adjs


# 60%的训练集，20%的验证集，20%的测试集
def get_evaluation_data(args, graphs):
    # np.random.seed(args.seed)
    edges_next = list(set(graphs[args.now + 1].edges()))
    edges_pos = []  # 正样本
    for e in edges_next:
        if e[0] < args.N and e[1] < args.N and [e[1], e[0]] not in edges_pos:  # 该边连接的两点若在上一时刻就出现则可以作为正样本
            edges_pos.append(list(e))
    edges_neg = []  # 负样本
    while len(edges_neg) < len(edges_pos):  # 正负采样数量相同
        idx_i = np.random.randint(0, args.N)  # 随机选择i,j节点
        idx_j = np.random.randint(0, args.N)
        if idx_i == idx_j:  # 忽略自连接
            continue
        if [idx_i, idx_j] in edges_pos or [idx_j, idx_i] in edges_pos:  # 忽略正采样的边
            continue
        if idx_i >= args.N or idx_j >= args.N:
            continue
        if edges_neg:
            if [idx_i, idx_j] in edges_neg or [idx_j, idx_i] in edges_neg:  # 该边已经存在于edges_neg中
                continue
        edges_neg.append([idx_i, idx_j])
        # 划分训练集，测试集，验证集
    train_edges_pos, test_pos, train_edges_neg, test_neg = \
        train_test_split(edges_pos, edges_neg, test_size=0.4)
    val_edges_pos, test_edges_pos, val_edges_neg, test_edges_neg = \
        train_test_split(test_pos, test_neg, test_size=0.5)
    return train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg


def getAdjList(args, graph):
    # 创建adj_list
    adj_list = []
    for _ in range(max(graph.nodes()) + 1):
        adj_list.append([])
    for link in graph.edges():
        adj_list[link[0]].append(link[1])
        adj_list[link[1]].append(link[0])
    args.N = args.number[args.now]  # 统计该张时间快照中的节点数量
    return adj_list
