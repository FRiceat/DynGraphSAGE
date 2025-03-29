import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
import math


class AttentionLayer(nn.Module):
    def __init__(self, args, features_dim, out_feat_dim, device):
        super(AttentionLayer, self).__init__()
        self.args = args
        self.features_dim = features_dim
        self.out_feat_dim = out_feat_dim
        self.device = device
        self.W = nn.Parameter(torch.empty(size=(self.features_dim, self.out_feat_dim)))
        nn.init.uniform_(self.W.data)
        self.W_attention = nn.Parameter(torch.empty(size=(self.args.out_dim, self.out_feat_dim)))
        nn.init.uniform_(self.W_attention.data)
        self.a = nn.Parameter(torch.empty(size=(self.out_feat_dim * 2, 1)))
        nn.init.uniform_(self.a.data)
        self.leakyrelu = nn.LeakyReLU(self.args.alpha)

    def forward(self, h, adj_list):
        mask = self.A_creat(adj_list)
        h = torch.mm(h, self.W)
        h1 = torch.matmul(h, self.a[:self.out_feat_dim, :])
        h2 = torch.matmul(h, self.a[self.out_feat_dim:, :])
        e = h1 + h2.T
        e = torch.mul(mask.to(self.device), e)
        e = self.leakyrelu(e)
        # e = F.dropout(e, 0.5, training=self.training)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(e != 0., e, zero_vec)
        # attention = F.softmax(attention / math.sqrt(self.out_feat_dim), dim=1)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, 0.1, training=self.training)
        feat = torch.matmul(attention, h)
        return F.elu(feat)
        # return feat

    def A_creat(self, adj):
        mask = Variable(torch.zeros(len(adj), len(adj)))
        column_indices = [n for neigh in adj for n in neigh]
        row_indices = [i for i in range(len(adj)) for _ in range(len(adj[i]))]
        for j in range(len(column_indices)):
            mask[row_indices[j], column_indices[j]] += 1
            # mask[row_indices[j], column_indices[j]] = 1

        num_mask = mask.sum(1, keepdim=True).to(self.device)
        mask = mask.to(self.device).div(num_mask)
        mask += torch.eye(len(adj)).to(self.device)
        num_mask = mask.sum(1, keepdim=True).to(self.device)
        mask = mask.to(self.device).div(num_mask)
        # mask += torch.eye(len(adj)).to(self.device)
        return mask
