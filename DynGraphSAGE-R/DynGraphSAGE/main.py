from __future__ import division
from __future__ import print_function
import argparse
import pickle as pkl
import numpy as np
import torch
import time
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from minibatch import MyDataset
from models import DynGraphSAGE
from utils import copy_data, load_data, getAdjList
import random
from sklearn.metrics.pairwise import cosine_similarity
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='act_mooc')  # 数据集名称
parser.add_argument('--N', type=int, default=0)  # 当前时间快照中有几个节点
parser.add_argument('--user_number', default=[])  # 各个时间快照中都有几个user节点
parser.add_argument('--item_number', default=[])  # 各个时间快照中都有几个item节点
parser.add_argument('--user_N', type=int, default=0)  # 当前时间快照中有几个user节点
parser.add_argument('--item_N', type=int, default=0)  # 当前时间快照中有几个item节点
parser.add_argument('--user_sum', type=int, default=0)  # user节点总数
parser.add_argument('--item_sum', type=int, default=0)  # item总数
parser.add_argument('--snapshot_num', default=1)  # 时间快照的数量
parser.add_argument('--number', default=[])  # 各个时间快照中都有几个节点
parser.add_argument('--now', type=int, default=0)  # 当前是第几张时间快照
parser.add_argument('--GPU_ID', type=int, nargs='?', default=0)  # GPU_ID
parser.add_argument('--seed', type=int, default=72)  # 随机数种子
parser.add_argument('--epochs', type=int, default=2)  # 训练次数
parser.add_argument('--lr', type=float, default=0.005)  # 初始学习率
parser.add_argument('--batch_size', type=int, default=512)  # 每次训练最多节点数
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--first_dim', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=128)  # 隐藏层维度
parser.add_argument('--out_dim', type=int, default=128)  # 输出层维度
parser.add_argument('--alpha', type=float, default=0.02, help='Alpha for the leaky_relu.')
parser.add_argument('--aggregate_1_num', type=int, default=10)  # 空间一聚邻居数
parser.add_argument('--aggregate_2_num', type=int, default=20)  # 空间二聚邻居数
parser.add_argument('--pos_num', type=int, default=20)  # 损失的正采样数量
parser.add_argument('--neg_num', type=int, default=20)  # 损失的负采样数量
parser.add_argument('--past', type=int, default=2)  # 时间维度邻居数量
parser.add_argument('--K', type=int, default=10)  # top_K
parser.add_argument('--stru_aggregate_type', type=str, default='mean')  # 空间维度聚合器类型
parser.add_argument('--time_aggregate_type', type=str, default='gcn')  # 时间维度聚合器类型
parser.add_argument('--no_stop', action='store_true', default=True)  # 是否继续
parser.add_argument('--hidden_info', action='store_true', default=True)  # 是否使用隐藏层信息

args = parser.parse_args()
device = torch.device("cuda:" + str(args.GPU_ID) if torch.cuda.is_available() else "cpu")  # GPU or CPU
print('device:', device)

# 配置日志记录器
logging.basicConfig(filename=str(args.data_name) + '_example.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 创建一个日志记录器
logger = logging.getLogger(__name__)

# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)
if args.now == 0:
    copy_data(args)

graphs = load_data(args)
while args.no_stop:
    adj_list = getAdjList(args, graphs[args.now])
    dataset = MyDataset(args, graphs[args.now], adj_list)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = DynGraphSAGE(args=args, graph=graphs[args.now], features_dim=args.first_dim, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)


    def train(best_loss, patient, agg_neigh_list1, agg_neigh_list2):
        global neighbor
        # torch.manual_seed(args.seed)
        loss_all = 0
        for _, feed_dict in enumerate(dataloader):
            optimizer.zero_grad()
            loss_train = getLoss(feed_dict, agg_neigh_list1, agg_neigh_list2)
            loss_all += loss_train
            loss_train.backward(retain_graph=False)
            optimizer.step()
            scheduler.step()
        if best_loss > loss_all:
            best_loss = loss_all
            patient = 0
            torch.save(model.state_dict(), '../embedding/%s/models/models_%s_%s_%s_%s.pth' % (
                args.data_name, args.stru_aggregate_type, args.time_aggregate_type, args.now, args.hidden_info))
        else:
            patient += 1

        # 遍历并输出模型参数
        # for name, param in relation_model.named_parameters():
        #     print("学习参数：" + str(name))
        #     # print(param.data)
        #
        # # 输出梯度，注意：在模型训练之前，梯度通常是None
        # for name, param in relation_model.named_parameters():
        #     if param.grad is not None:
        #         print(f"学习参数 {name} 的梯度为: \n{param.grad.data}\n\n")
        return best_loss, patient


    def getLoss(feed_dict, agg_neigh_list1, agg_neigh_list2):
        node_source, node_pos, node_neg = feed_dict.values()
        node_source = node_source[0].reshape(-1)
        node_pos = node_pos[0]
        node_neg = node_neg[0]
        embes_hidden1, embes_hidden2, embes = model.forward(graphs, agg_neigh_list1, agg_neigh_list2)
        if embes == 'null':
            embes = embes_hidden2
        source_tensor = embes[node_source, :].unsqueeze(1)
        pos_tensor = embes[node_pos, :]
        neg_tensor = embes[node_neg, :]
        loss = -F.logsigmoid(torch.matmul(source_tensor, pos_tensor.transpose(1, 2)).squeeze(1)).sum()
        loss -= F.logsigmoid(- torch.matmul(source_tensor, neg_tensor.transpose(1, 2)).squeeze(1)).sum()
        losses = loss / (args.N * args.pos_num)
        return losses


    def aggregate_neighbors_sample(adj_list, aggregate_num):
        # random.seed(args.seed)
        # np.random.seed(args.seed)
        aggregate_neighbors = []
        for adj in adj_list:
            if len(adj) != 0:
                sampled_indices = np.random.choice(len(adj), size=aggregate_num, replace=True)
                sampled_arrays = [adj[idx] for idx in sampled_indices]
                aggregate_neighbors.append(sampled_arrays)
            else:
                aggregate_neighbors.append([-1])
        return aggregate_neighbors


    def recommend_top_k(user_index, scores, graph, k):
        ndcg10 = []
        ground_truth = list(set(np.array(graph.graph['test_pos'][user_index])))
        top_k_indices = np.argsort(scores[user_index])[::-1][:k] + args.user_N
        hits = np.intersect1d(ground_truth, top_k_indices)
        recall = len(hits) / len(ground_truth) if len(ground_truth) > 0 else 'null'
        if len(ground_truth) > 0:
            for rank in range(len(top_k_indices)):
                if top_k_indices[rank] in ground_truth:
                    ndcg10.append(1 / np.log2(rank + 2))
            if len(ndcg10) == 0:
                ndcg10 = 'null'
            else:
                ndcg10 = float(np.mean(ndcg10))
        else:
            ndcg10 = 'null'
        return recall, ndcg10


    best_loss = 9e15
    patient = 0
    recall_log = ['recall']
    AUC_log = []
    agg_neigh_list1 = aggregate_neighbors_sample(adj_list, args.aggregate_1_num)
    agg_neigh_list2 = aggregate_neighbors_sample(adj_list, args.aggregate_2_num)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.001)
    for epoch in range(args.epochs):
        # print("当前是第" + str(args.now) + "张时间快照，正在训练第" + str(epoch) + "次")
        if patient < 20 or epoch < 50:
            model.train()
            loss_epoch, patient = train(best_loss, patient, agg_neigh_list1, agg_neigh_list2)
            # recall_log.append("{:.5f}".format(recall_epoch))
            best_loss = loss_epoch
        else:
            print("运行了" + str(epoch) + "次")
            break

    model.load_state_dict(
        torch.load('../embedding/%s/models/models_%s_%s_%s_%s.pth' % (
            args.data_name, args.stru_aggregate_type, args.time_aggregate_type, args.now, args.hidden_info)))
    model.eval()
    with torch.no_grad():
        embes_hidden1, embes_hidden2, embedding = model(graphs, agg_neigh_list1, agg_neigh_list2)
        graph = graphs[args.now]
        graph.hidden1 = embes_hidden1
        graph.hidden2 = embes_hidden2
        if embedding == 'null':
            graph.embedding = embes_hidden2
            embedding = embes_hidden2

        embedding = embedding.data.cpu().numpy()
        user_item_scores = cosine_similarity(embedding[:args.user_N], embedding[args.user_N:])
        all_recalls_10 = []
        all_ndcg_10 = []
        for user_index in range(args.user_N):
            if len(list(set(np.array(graph.graph['test_pos'][user_index])))) != 0:
                recall, ndcg = recommend_top_k(user_index, user_item_scores, graph, 10)
                if recall != 'null':
                    all_recalls_10.append(recall)
                if ndcg != 'null':
                    all_ndcg_10.append(ndcg)
        recall_10 = np.mean(all_recalls_10)
        ndcg_10 = np.mean(all_ndcg_10)

        all_recalls_20 = []
        all_ndcg_20 = []
        for user_index in range(args.user_N):
            if len(list(set(np.array(graph.graph['test_pos'][user_index])))) != 0:
                recall, ndcg = recommend_top_k(user_index, user_item_scores, graph, 20)
                if recall != 'null':
                    all_recalls_20.append(recall)
                if ndcg != 'null':
                    all_ndcg_20.append(ndcg)
        recall_20 = np.mean(all_recalls_20)
        ndcg_20 = np.mean(all_ndcg_20)

        print("now:", args.now)
        print("recall_10", recall_10)
        print("ndcg_10", ndcg_10)
        print("recall_20", recall_20)
        print("ndcg_20", ndcg_20)
        print("----------------------")
        # print("runtime:", runtime)
        # 开始记录日志
        info_list = ['data_name:' + str(args.data_name) + ',now:' + str(args.now) + ',struc:' + str(
            args.stru_aggregate_type) + ',time:' + args.time_aggregate_type + ',hidden_info:' + str(args.hidden_info)]
        logger.info(info_list)
        # loss变化记录
        logger.info(recall_log)
        # 实验结果记录
        if AUC_log:
            logger.info(AUC_log)
        args.now = args.now + 1  # 准备进入下一张时间快照
        if args.now > args.snapshot_num - 1:
            args.no_stop = False

save_path = "../embedding/%s/%s_%s_%s_%s.pkl" % (
    args.data_name, args.data_name, args.stru_aggregate_type, args.time_aggregate_type, args.hidden_info)
with open(save_path, "wb") as f:
    pkl.dump(graphs, f)
