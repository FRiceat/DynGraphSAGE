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
from utils import copy_data, load_data, get_evaluation_data, getAdjList
from link_prediction import evaluate_classifier
import random
import logging

start = time.perf_counter()
parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='uci')  # 数据集名称
parser.add_argument('--N', type=int, default=0)  # 当前时间快照中有几个节点
parser.add_argument('--snapshot_num', default=1)  # 时间快照的数量
parser.add_argument('--number', default=[])  # 各个时间快照中都有几个节点
parser.add_argument('--now', type=int, default=0)  # 当前是第几张时间快照
parser.add_argument('--GPU_ID', type=int, nargs='?', default=1)  # GPU_ID
parser.add_argument('--seed', type=int, default=2024)  # 随机数种子
parser.add_argument('--epochs', type=int, default=200)  # 训练次数
parser.add_argument('--lr', type=float, default=0.005)  # 初始学习率
parser.add_argument('--batch_size', type=int, default=512)  # 每次训练最多节点数
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--first_dim', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=128)  # 隐藏层维度
parser.add_argument('--out_dim', type=int, default=128)  # 输出层维度
parser.add_argument('--alpha', type=float, default=0.01, help='Alpha for the leaky_relu.')
parser.add_argument('--aggregate_1_num', type=int, default=10)  # 空间一聚邻居数
parser.add_argument('--aggregate_2_num', type=int, default=20)  # 空间二聚邻居数
parser.add_argument('--pos_num', type=int, default=20)  # 损失的正采样数量
parser.add_argument('--neg_num', type=int, default=20)  # 损失的负采样数量
parser.add_argument('--past', type=int, default=2)  # 时间维度邻居数量
parser.add_argument('--stru_aggregate_type', type=str, default='lstm')  # 空间维度聚合器类型
parser.add_argument('--time_aggregate_type', type=str, default='mean')  # 时间维度聚合器类型
parser.add_argument('--no_stop', action='store_true', default=True)  # 是否继续
parser.add_argument('--hidden_info', action='store_true', default=False)  # 是否使用隐藏层信息
parser.add_argument('--biaoshi', type=int, default=0)  # 标识

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
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
if args.now == 0:
    copy_data(args)

graphs, adjs = load_data(args)
while args.no_stop:
    adj = adjs[args.now]
    adj_list = getAdjList(args, graphs[args.now])
    dataset = MyDataset(args, graphs[args.now], adj_list)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = DynGraphSAGE(args=args, graph=graphs[args.now], features_dim=args.number[-1], device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)


    def train(best_loss, patient, agg_neigh_list1, agg_neigh_list2):
        # torch.manual_seed(args.seed)
        loss_all = 0
        for _, feed_dict in enumerate(dataloader):
            optimizer.zero_grad()
            loss_train = getLoss(feed_dict, agg_neigh_list1, agg_neigh_list2)
            loss_all += loss_train
            loss_train.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
        if loss_all < best_loss:
            best_loss = loss_all
            patient = 0
            torch.save(model.state_dict(), '../embedding/%s/models/models_%s_%s_%s_%s.pth' % (
                args.data_name, args.stru_aggregate_type, args.time_aggregate_type, args.now, args.hidden_info))
        else:
            patient += 1

        # # 遍历并输出模型参数
        # for name, param in model.named_parameters():
        #     print("学习参数：" + str(name))
        #     # print(param.data)
        #
        # # 输出梯度，注意：在模型训练之前，梯度通常是None
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"学习参数 {name} 的梯度为: \n{param.grad.data}\n\n")
        return best_loss, patient


    def getLoss(feed_dict, agg_neigh_list1, agg_neigh_list2):
        node_source, node_pos, node_neg = feed_dict.values()
        node_source = node_source[0].reshape(-1)
        node_pos = node_pos[0]
        node_neg = node_neg[0]
        embes_hidden1, embes_hidden2, embes, h_n1, c_n1, h_n2, c_n2 = model.forward(graphs, adj_list, agg_neigh_list1,
                                                                                    agg_neigh_list2)
        if embes == 'null':
            embes = embes_hidden2
        source_tensor = embes[node_source, :].unsqueeze(1)
        pos_tensor = embes[node_pos, :]
        neg_tensor = embes[node_neg, :]
        pos_loss = -F.logsigmoid(
            torch.matmul(source_tensor, pos_tensor.transpose(1, 2)).squeeze(1)).sum() / args.pos_num
        neg_loss = -F.logsigmoid(
            torch.matmul(-source_tensor, neg_tensor.transpose(1, 2)).squeeze(1)).sum() / args.neg_num
        losses = (pos_loss + neg_loss) / args.N

        # 时间维度与近邻产生的损失
        if args.now >= 2:
            num = args.number[args.now - args.past]
            embedding1 = graphs[args.now - 1].embedding[:num]
            embedding2 = embes[:num]
            time_loss = torch.div(-F.logsigmoid(torch.mul(embedding1, embedding2).sum(-1)).sum(), num)
            losses += time_loss
        # print('损失：', losses)
        return losses


    def aggregate_neighbors_sample(adj_list, aggregate_num):
        # random.seed(args.seed)
        # np.random.seed(args.seed)
        adj_list = [list(set(neighbors)) for neighbors in adj_list]
        # adj_list = [neighbors for neighbors in adj_list]
        aggregate_neighbors = [np.random.choice(to_neigh, aggregate_num, replace=False) if len(
            to_neigh) >= aggregate_num else np.random.choice(to_neigh, aggregate_num, replace=True) for to_neigh in
                               adj_list]
        # aggregate_neighbors = [random.sample(to_neigh, aggregate_num) if len(to_neigh) >= aggregate_num
        #                        else np.random.choice(to_neigh, aggregate_num, replace=True) for to_neigh in adj_list]
        return aggregate_neighbors


    # 运行函数
    best_loss = 9e15
    patient = 0
    loss_log = ['loss']
    agg_neigh_list1 = aggregate_neighbors_sample(adj_list, args.aggregate_1_num)
    agg_neigh_list2 = aggregate_neighbors_sample(adj_list, args.aggregate_2_num)
    AUC_log = []
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.001)
    for epoch in range(args.epochs):
        # print("当前是第" + str(args.now) + "张时间快照，正在训练第" + str(epoch) + "次")

        model.train()
        loss_epoch, patient = train(best_loss, patient, agg_neigh_list1, agg_neigh_list2)
        loss_log.append("{:.5f}".format(loss_epoch.cpu().item()))
        if loss_epoch < best_loss:
            best_loss = loss_epoch
    if args.now == 0:
        model.load_state_dict(
            torch.load('../embedding/%s/models/models_%s_%s_%s_%s.pth' % (
                args.data_name, args.stru_aggregate_type, args.time_aggregate_type, args.now, args.hidden_info)))
    else:
        model.load_state_dict(
            torch.load('../embedding/%s/models/models_%s_%s_%s_%s.pth' % (
                args.data_name, args.stru_aggregate_type, args.time_aggregate_type, args.now, args.hidden_info)))
    model.eval()
    with torch.no_grad():
        embes_hidden1, embes_hidden2, embedding, h_n1, c_n1, h_n2, c_n2 = model(graphs, adj_list, agg_neigh_list1,
                                                                                agg_neigh_list2)
        graph = graphs[args.now]
        graph.hidden1 = embes_hidden1
        graph.hidden2 = embes_hidden2
        graph.h_t1 = h_n1
        graph.h_t2 = h_n2
        graph.c_t1 = c_n1
        graph.c_t2 = c_n2
        if embedding == 'null':
            graph.embedding = embes_hidden2
            embedding = embes_hidden2.data.cpu().numpy()
        else:
            graph.embedding = embedding
            embedding = embedding.data.cpu().numpy()
        if int(args.now) >= 2:
            train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg \
                = get_evaluation_data(args, graphs)
            val_result, test_result, _, _ = evaluate_classifier(train_edges_pos, train_edges_neg, val_edges_pos,
                                                                val_edges_neg, test_edges_pos, test_edges_neg,
                                                                embedding,
                                                                embedding, args)
            auc_val = val_result["HAD"][1]
            auc_test = test_result["HAD"][1]
            print("Val AUC {:.5f} Test AUC {:.5f}".format(auc_val, auc_test))
            AUC_log.append("Val AUC {:.5f} Test AUC {:.5f}".format(auc_val, auc_test))
        end = time.perf_counter()
        runtime = end - start
        # print("runtime:", runtime)
        save_path = "../embedding/%s/%s_%s_%s_%s.pkl" % (
            args.data_name, args.data_name, args.stru_aggregate_type, args.time_aggregate_type, args.hidden_info)
        with open(save_path, "wb") as f:
            pkl.dump(graphs, f)
        # 开始记录日志
        info_list = ['data_name:' + str(args.data_name) + ',now:' + str(args.now) + ',struc:' + str(
            args.stru_aggregate_type) + ',time:' + args.time_aggregate_type + ',hidden_info:' + str(args.hidden_info)]
        logger.info(info_list)
        # loss变化记录
        logger.info(loss_log)
        # 实验结果记录
        if AUC_log:
            logger.info(AUC_log)
        args.now = args.now + 1  # 准备进入下一张时间快照
        if args.now >= args.snapshot_num - 1:
            args.no_stop = False
