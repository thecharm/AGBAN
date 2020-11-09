# -*- coding: utf-8 -*-
# @Author: Zhiwei Wu
# @Date: 2019/9/29 17:07
# @contact: zhiwei.w@qq.com

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.crf import CRF
from model.bilinearattention import BiAttention
from model.fc import FCNet
from model.bc import BCNet

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class AGBAN(nn.Module):
    """

    """

    def __init__(self, data):
        super(AGBAN, self).__init__()
        print('build AGBAN network...')
        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        self.bilstm_flag = data.HP_bilstm

        self.ner_loss_lambda = data.ner_loss_lambda
        self.v_loss_lambda = data.v_loss_lambda
        self.t_loss_lambda = data.t_loss_lambda

        #
        self.v_dim = 200
        self.t_dim = data.HP_hidden_dim  # 200
        self.c_dim = 200

        # attention HP
        self.glimpse = data.glimpse

        # word represent = char lstm represent + word emb
        self.char_embeddings = nn.Embedding(data.char_alphabet.size(), data.char_emb_dim)
        if data.pretrain_char_embedding is not None:
            self.char_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_char_embedding))
        else:
            self.char_embeddings.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.char_alphabet.size(), data.char_emb_dim)))
        self.char_drop = nn.Dropout(data.HP_dropout)  # emb -> dropout -> char lstm
        self.char_lstm = nn.LSTM(data.char_emb_dim, data.HP_char_hidden_dim // 2, num_layers=1, batch_first=True,
                                 bidirectional=True)

        self.word_embeddings = nn.Embedding(data.word_alphabet.size(), data.word_emb_dim)
        if data.pretrain_word_embedding is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embeddings.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.word_alphabet.size(), data.word_emb_dim)))
        self.word_drop = nn.Dropout(data.HP_dropout)  # [char_presentation, word embedding] -> dropout

        # word seq lstm
        self.input_size = data.word_emb_dim + data.HP_char_hidden_dim
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim
        self.word_lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=data.HP_lstm_layer, batch_first=True,
                                 bidirectional=data.HP_bilstm)
        self.droplstm = nn.Dropout(data.HP_dropout)  # word seq lstm out -> dropout

        # Bilinear attention
        self.v_att = BiAttention(self.v_dim, self.t_dim, self.c_dim, self.glimpse)  # 200-> 200*3 BiAttention内部写死 映射到3xc_dim
        b_net = []  # 根据 v_feats 和 attention的得分算出 每个词对应的v_feats

        g_t_prj = []  # Gated project for text
        g_v_prj = []  # Gated project for visual
        g_prj = []  # Gated project for [text, visual]
        for i in range(self.glimpse):
            b_net.append(FCNet([self.v_dim, self.c_dim], act='ReLU', dropout=0.2))  # return b x h_out x v x q
            g_t_prj.append(FCNet([self.t_dim, self.c_dim], ''))
            g_v_prj.append(FCNet([self.v_dim, self.c_dim], ''))
            g_prj.append(FCNet([self.t_dim+self.v_dim, self.c_dim], ''))

        self.b_net = nn.ModuleList(b_net)
        self.g_t_prj = nn.ModuleList(g_t_prj)
        self.g_v_prj = nn.ModuleList(g_v_prj)
        self.g_prj = nn.ModuleList(g_prj)

        self.softmax = nn.Softmax(dim=2)

        # bn
        self.cat_bn = nn.BatchNorm1d(data.HP_hidden_dim)

        # hidden to tag, add the start and end tag
        self.hidden2tag = nn.Linear(data.HP_hidden_dim, data.label_alphabet_size + 2)

        # crf
        self.crf = CRF(data.label_alphabet_size, self.gpu)

        # visual feature map to the text space and keep the same distribution
        self.map_to_text = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.v_dim),
        )

        # modal clf
        self.modal_clf = nn.Sequential(
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, 2),
        )

        # criter
        textual_weight = 4000 * data.max_obj / (64439 + 4000 * data.max_obj)
        visual_weight = 64439 / (64439 + 4000 * data.max_obj)
        weight = torch.FloatTensor([visual_weight, textual_weight])
        self.modal_clf_loss = nn.CrossEntropyLoss(ignore_index=-1, weight=weight, reduction='sum')

        # save textual and visual modal feat
        self.textual_feats = None
        self.visual_feats = None
        self.att = None
        self.logits = None

    def _get_lstm_features(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                           char_seq_recover, vfeature_input,  v_mask, alpha):
        """
        char-> char embedding -> dropout -> char lstm -> char representation
            word -> word embedding
            [char representation, word embedding] -> dropout -> word lstm -> dropout -> Linear -> tagscores
        :param word_inputs: (batch_size, sent_len)
        :param feature_inputs:  [(batch_size, sent_len), ...] list of variables
        :param word_seq_lengths: list of batch_size, (batch_size, 1)
        :param char_inputs: (batch_size * sent_len, word_length)
        :param char_seq_lengths: (batch_size * sent_len, 1)
        :param char_seq_recover: variable which records the char order information, used to recover char order
        :param vfeature_input: tensor(batch_size, 1024)
        :param v_mask: (batch_size, obj)
        :return:
            variable(batch_size,sent_len, hidden_dim)
        """
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)
        char_batch_size = char_inputs.size(0)
        obj = v_mask.size(1)

        ## 1. build visual and text feature
        # char -> emb -> drop
        char_embeds = self.char_drop(
            self.char_embeddings(char_inputs))  # (batch_size * sent_len, word_length, char_emb_dim)
        char_hidden = None
        # -> char lstm,
        pack_char_input = pack_padded_sequence(char_embeds, char_seq_lengths.cpu().numpy(), batch_first=True)
        char_rnn_out, char_hidden = self.char_lstm(pack_char_input, char_hidden)
        char_features = char_hidden[0].transpose(1, 0).contiguous().view(char_batch_size,
                                                                         -1)  # (batch_size * sent_len, char_hidden_dim)
        char_features = char_features[char_seq_recover]
        # cat char_hidden_dim for every char in a word
        char_features = char_features.view(batch_size, sent_len, -1)  # (batch_size, sent_len, char_hidden_dim)

        # word -> word emb
        word_embs = self.word_embeddings(word_inputs)
        # concat-> word represent
        word_represent = torch.cat([word_embs, char_features], 2)
        word_represent = self.word_drop(word_represent)  # (batch_size, sent_len, char_hidden_dim + word_emb_dim)
        # -> word seq lstm
        packed_word = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.word_lstm(packed_word, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        t_feats = self.droplstm(lstm_out)  # (batch_size, sent_len, hidden_dim)

        # reshape visual feature for BN (b*obj, C)
        vfeature_input = vfeature_input.view(batch_size, -1, 1024)  # (b, obj, feature_dim)
        vfeature_input = vfeature_input.view(-1, 1024)  # （obj*b, feature_dim)
        v_feats = self.map_to_text(vfeature_input)  # (obj*b, 200)
        v_feats = v_feats.view(batch_size, -1, 200) # (b, v, 200)
        # save textual feats and visual feats
        self.textual_feats = t_feats.contiguous().view(-1, 200)
        self.visual_feats = v_feats.contiguous().view(-1, 200)

        # domain clf
        # text clf
        textual_clf_input = t_feats.contiguous().view(-1, 200)  # (b*seq_len, 200)
        revers_textual_clf_input = ReverseLayerF.apply(textual_clf_input, alpha)
        textual_clf_outputs = self.modal_clf(revers_textual_clf_input)  # (b*seq_len, 2)
        # visual clf
        visual_clf_input = v_feats.contiguous().view(-1, 200)
        revers_cor_vfeature = ReverseLayerF.apply(visual_clf_input, alpha)
        visual_clf_outputs = self.modal_clf(revers_cor_vfeature)  # (b*obj, 2)

        # Biliear attention
        # input: t_feats(b, t, d), v_feats(b, v, d)
        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v_feats, t_feats, v_mask) # b x g x v x t
        for g in range(self.glimpse):
            # multi-head attention visual feats be added to t_feats
            att_g = att[:, g, :, :].transpose(1, 2)  # b x t x v
            v_g = self.b_net[g](v_feats)  # b x v x d
            b_emb[g] = torch.bmm(att_g, v_g)  # b x t x d

            # Gated
            g_t = self.g_t_prj[g](t_feats)  # b x t x d
            g_v = self.g_v_prj[g](b_emb[g])  # b x t x d
            gate = torch.sigmoid(self.g_prj[g](torch.cat([g_t, g_v], 2)))
            # fusion Gated attention visual feats and textual feats
            t_feats = gate * b_emb[g] + t_feats  # b x t x d

        # save att logits
        self.att = att
        self.logits = logits

        # BN layer
        final_feature = t_feats.transpose(2, 1).contiguous()  # (b, d, t)
        final_feature = self.cat_bn(final_feature)
        final_feature = final_feature.transpose(2, 1).contiguous()
        # -> tagscore
        outputs = self.hidden2tag(final_feature)
        return outputs, textual_clf_outputs, visual_clf_outputs

    def calculate_loss(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, \
                       char_seq_recover, batch_label, mask, vfeature_input, v_modal_label, t_modal_label, v_mask, alpha=0):
        """
            char-> char embedding -> dropout -> char lstm -> char representation
            word -> word embedding
            [char representation, word embedding] -> dropout -> word lstm -> dropout -> Linear -> tagscores
            - crf -> loss
        :param word_inputs: (batch_size, sent_len)
        :param feature_inputs:  [(batch_size, sent_len), ...] list of variables
        :param word_seq_lengths: list of batch_size, (batch_size, 1)
        :param char_inputs: (batch_size * sent_len, word_length)
        :param char_seq_lengths: (batch_size * sent_len, 1)
        :param char_seq_recover: variable which records the char order information, used to recover char order
        :param batch_label: (batch_size, sent_len)
        :param mask: (batch_size, sent_len)
        :param vfeature_input: (batch_size, 1024)
        :return:
            variable(batch_size,sent_len, hidden_dim)
        """
        outs, textual_clf_outputs, visual_clf_outputs = self._get_lstm_features(word_inputs, feature_inputs,
                                                                                word_seq_lengths, char_inputs,
                                                                                char_seq_lengths, char_seq_recover,
                                                                                vfeature_input, v_mask, alpha)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        # crf
        ner_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
        scores, tag_seq = self.crf._viterbi_decode(outs, mask)  # batch,
        if self.average_batch:
            ner_loss = ner_loss / batch_size

        # modal clf loss
        # visual_clf_outputs: (obj*b, 2) mask (b, obj) ->
        visual_loss = self.modal_clf_loss(visual_clf_outputs, v_modal_label)
        textual_loss = self.modal_clf_loss(textual_clf_outputs, t_modal_label)

        total_loss = self.ner_loss_lambda * ner_loss + self.v_loss_lambda * visual_loss \
                     + self.t_loss_lambda * textual_loss

        return total_loss, \
               self.ner_loss_lambda * ner_loss, \
               self.t_loss_lambda * textual_loss, \
               self.v_loss_lambda * visual_loss, \
               tag_seq, textual_clf_outputs, visual_clf_outputs

    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,
                mask, vfeature_input, v_modal_label, t_modal_label, v_mask, alpha=0):
        outs, textual_clf_outputs, visual_clf_outputs = self._get_lstm_features(word_inputs, feature_inputs,
                                                                                word_seq_lengths, char_inputs,
                                                                                char_seq_lengths, char_seq_recover,
                                                                                vfeature_input, v_mask, alpha)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        return tag_seq

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def decode_nbest(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                     char_seq_recover, mask, nbest):
        outs = self._get_lstm_features(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                       char_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        scores, tag_seq = self.crf._viterbi_decode_nbest(outs, mask, nbest)
        return scores, tag_seq
