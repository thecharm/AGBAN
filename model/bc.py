"""
Bilinear Attention Networks
Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
https://arxiv.org/abs/1805.07932

This code is written by Jin-Hwa Kim.
"""
from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from model.fc import FCNet


class BCNet(nn.Module):
    """Simple class for non-linear bilinear connect network
    """

    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=[.2, .5], k=3):
        super(BCNet, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out  # maybe glimpse == 8

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])  # 全连接层 v_dim -> h_dim*k
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])  # attention
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if None == h_out:
            pass
        # h_mat, h_bias , h_net 即论文中的参数矩阵 P
        elif h_out <= self.c:
            # h_out <= 32
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())  # 1 x h_out x 1 x h_dim *k
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim, h_out), dim=None)

    def forward(self, v, q):
        if None == self.h_out:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)  # b, len, v_dim -> b, len, h_dim*k -> b, h_dim*k, v, 1
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)  # b, len, q_dim -> b, len, h_dim*k -> b, h_dim*k, 1, q
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            logits = d_.transpose(1, 2).transpose(2, 3)  # b x v x q x h_dim
            return logits

        # broadcast Hadamard product, matrix-matrix production
        # fast computation but memory inefficient
        # epoch 1, time: 157.84
        # for attention modulor: h_out = glimpse = 8, self.c = 32
        # for attention
        # v: [batch, num_objs, obj_dim]
        # q: [batch, q_len, q_dim] q_dim == h_dim == 1280

        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v)).unsqueeze(1)  # b x 1 x v x h_dim
            q_ = self.q_net(q)  # b x q x h_dim
            h_ = v_ * self.h_mat  # broadcast, b x h_out x v x h_dim
            logits = torch.matmul(h_, q_.unsqueeze(1).transpose(2, 3))  # b x h_out x v x q
            logits = logits + self.h_bias
            return logits  # b x h_out x v x q

        # batch outer product, linear projection
        # memory efficient but slow computation
        # epoch 1, time: 304.87
        else:
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)  # b x h_dim x v x 1
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)  # b x h_dim x 1 x q
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            return logits.transpose(2, 3).transpose(1, 2)  # b x h_out(glimpse) x v x q

    def forward_with_weights(self, v, q, w):
        """
        BA network：single layer
        1. visual project hidden dim
        2. visual times attention(w)
        :param v:
        :param q:
        :param w: attention (b, 1, v, q)
        :return:
        """
        v_ = self.v_net(v).transpose(1,2).unsqueeze(2)  # b x d x 1 x v
        q_ = self.q_net(q).transpose(1,2).unsqueeze(3)  # b x d x q x 1
        logits = torch.matmul(torch.matmul(v_, w.unsqueeze(1)), q_)  # b x d x 1 x 1

        logits = logits.squeeze(3).squeeze(2)  # (b, d)
        if 1 < self.k:
            logits = logits.unsqueeze(1)  # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k  # sum-pooling
        return logits


if __name__ == '__main__':
    # net = BCNet(1024, 1024, 1024, 1024).cuda()
    # x = torch.Tensor(512, 36, 1024).cuda()
    # y = torch.Tensor(512, 14, 1024).cuda()
    # out = net.forward(x, y)

    net = BCNet(1024, 1024, 1024, None, k=1).cuda()
    v = torch.Tensor(512, 36, 1024).cuda()
    q = torch.Tensor(512, 14, 1024).cuda()
    att = torch.Tensor(512, 36, 14).cuda()
    out = net.forward_with_weights(v, q, att)
