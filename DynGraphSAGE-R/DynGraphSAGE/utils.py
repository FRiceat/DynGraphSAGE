import pickle as pkl
import shutil
import torch


def copy_data(args):
    shutil.copyfile("../data/%s/%s.pkl" % (args.data_name, args.data_name),
                    "../embedding/%s/%s.pkl" % (args.data_name, args.data_name))


def load_data(args):
    # torch.manual_seed(args.seed)
    with open("../embedding/%s/%s.pkl" % (args.data_name, args.data_name), "rb") as f:
        graphs = pkl.load(f)  # 加载图
    args.snapshot_num = len(graphs)  # 统计图中时间快照的数量
    for graph in graphs:  # 统计各张时间快照中节点数量
        args.number.append(graph.graph['adj'].shape[0] + graph.graph['adj'].shape[1])
        args.user_number.append(graph.graph['adj'].shape[0])
        args.item_number.append(graph.graph['adj'].shape[1])
    args.user_sum = graphs[-1].graph['adj'].shape[0]
    args.item_sum = graphs[-1].graph['adj'].shape[1]
    return graphs


def getAdjList(args, graph):
    # 创建adj_list
    args.N = graph.graph['adj'].shape[0] + graph.graph['adj'].shape[1]
    args.user_N = graph.graph['adj'].shape[0]
    args.item_N = graph.graph['adj'].shape[1]
    return graph.graph['train_pos']
