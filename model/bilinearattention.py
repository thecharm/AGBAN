"""
Bilinear Attention Networks
Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
https://arxiv.org/abs/1805.07932

This code is written by Jin-Hwa Kim.
"""
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from model.fc import FCNet
from model.bc import BCNet


class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2,.5]):

        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        # v_dim, q_dim, hid_dim, glimpse
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3), name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        p, logits = self.forward_all(v, q, v_mask)
        return p, logits

    def forward_all(self, v, q, v_mask):
        """

        :param v: [batch, num_objs, obj_dim]
        :param q: [batch, q_len, q_dim]
        :param v_mask: [batch, v]
        :return:
        """
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v, q)  # b x glimpse x v x q

        # 视觉padding部分的概率置为0
        mask = (1 - v_mask.int()).bool()
        mask = mask.unsqueeze(1).unsqueeze(3).expand(logits.size())
        logits.data.masked_fill_(mask.data, -1e9)

        # 如果 某个sample的视觉对象为0个，则得到的概率值为0
        p = nn.functional.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
        p = p.view(-1, self.glimpse, v_num, q_num)
        p = p.masked_fill(mask.data, 0)
        return p, logits
