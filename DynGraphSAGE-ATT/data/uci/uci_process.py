from collections import defaultdict
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from scipy.sparse import csr_matrix
import pickle as pkl

links = []
ts = []
node_cnt = 0
node_idx = {}
idx_node = []
with open('F:/DynGraphSAGE/data/uci/opsahl-ucsocial.edges') as f:
    lines = f.read().splitlines()
    for i in lines:
        if i[0] == '%':
            continue
        _, _, _, t = map(int, i.split(','))
        timestamp = datetime.fromtimestamp(t)
        ts.append(timestamp)
    START_DATE = min(ts) + timedelta(5)
    END_DATE = max(ts) - timedelta(60)
    for line in lines:
        if line[0] == '%':
            continue
        x, y, e, t = map(int, line.split(','))
        t = datetime.fromtimestamp(t)
        if t < START_DATE or t > END_DATE:
            continue
        else:
            if x not in node_idx:
                node_idx[x] = node_cnt
                node_cnt += 1
            if y not in node_idx:
                node_idx[y] = node_cnt
                node_cnt += 1
        links.append((node_idx[x], node_idx[y], t))
links.sort(key=lambda link: link[2])  # 排序

SLICE_DAYS = 10
slice_links = defaultdict(lambda: nx.MultiGraph())
slice_features = defaultdict(lambda: {})
slice_id = 0
for (a, b, time) in links:
    prev_slice_id = slice_id
    datetime_object = time
    days_diff = (datetime_object - START_DATE).days
    slice_id = days_diff // SLICE_DAYS
    if slice_id == 1 + prev_slice_id and slice_id > 0:
        slice_links[slice_id] = nx.MultiGraph()
        slice_links[slice_id].add_nodes_from(slice_links[slice_id - 1].nodes(data=True))
        assert (len(slice_links[slice_id].edges()) == 0)
    if slice_id == 1 + prev_slice_id and slice_id == 0:
        slice_links[slice_id] = nx.MultiGraph()
    if a not in slice_links[slice_id]:
        slice_links[slice_id].add_node(a)
    if b not in slice_links[slice_id]:
        slice_links[slice_id].add_node(b)
    slice_links[slice_id].add_edge(a, b, date=datetime_object)

onehot = np.identity(slice_links[max(slice_links.keys())].number_of_nodes())  # 最后一个图中的所有节点; 为每个节点建立one-hot向量
graphs = []
for id, slice in slice_links.items():
    tmp_feature = []
    for node in slice.nodes():  # 该图中节点
        tmp_feature.append(onehot[node])
    slice.graph["feature"] = csr_matrix(tmp_feature)  # 稀疏矩阵; 添加图中特征
    graphs.append(slice)  # 将图保存在list中

save_path = "F:/DynGraphSAGE/data/uci/uci.pkl"
with open(save_path, "wb") as f:
    pkl.dump(graphs, f)
