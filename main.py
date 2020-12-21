from __future__ import print_function
import time
import sys
import argparse
import random
import torch
import gc
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure
from model.seqlabel import SeqLabel
from model.sentclassifier import SentClassifier
from utils.data import Data
import model as models


try:
    import cPickle as pickle
except ImportError:
    import pickle

# seed_num = 42
# random.seed(seed_num)
# torch.manual_seed(seed_num)
# np.random.seed(seed_num)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def data_initialization(data):
    data.initial_feature_alphabets()
    data.build_alphabet(data.train_dir)
    data.build_alphabet(data.dev_dir)
    data.build_alphabet(data.test_dir)
    data.fix_alphabet()


def predict_check(pred_variable, gold_variable, mask_variable, sentence_classification=False):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    if sentence_classification:
        # print(overlaped)
        # print(overlaped*pred)
        right_token = np.sum(overlaped)
        total_token = overlaped.shape[0]  ## =batch_size
    else:
        right_token = np.sum(overlaped * mask)
        total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover,
                  sentence_classification=False):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    if sentence_classification:
        pred_tag = pred_variable.cpu().data.numpy().tolist()
        gold_tag = gold_variable.cpu().data.numpy().tolist()
        pred_label = [label_alphabet.get_instance(pred) for pred in pred_tag]
        gold_label = [label_alphabet.get_instance(gold) for gold in gold_tag]
    else:
        seq_len = gold_variable.size(1)
        mask = mask_variable.cpu().data.numpy()
        pred_tag = pred_variable.cpu().data.numpy()
        gold_tag = gold_variable.cpu().data.numpy()
        batch_size = mask.shape[0]
        pred_label = []
        gold_label = []
        for idx in range(batch_size):
            pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            assert (len(pred) == len(gold))
            pred_label.append(pred)
            gold_label.append(gold)
    return pred_label, gold_label


def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
    """
    # exit(0)
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    nbest = pred_variable.size(2)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = []
        for idz in range(nbest):
            each_pred = [label_alphabet.get_instance(pred_tag[idx][idy][idz]) for idy in range(seq_len) if
                         mask[idx][idy] != 0]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def evaluate(data, model, name, nbest=None):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
        exit(1)
    right_token = 0
    whole_token = 0
    nbest_pred_results = []
    pred_scores = []
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()
    batch_size = data.HP_batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1
    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask, batch_vfeature, v_modal_label, t_modal_label, v_mask = batchify_with_label(
            instance, data.HP_gpu, False, data.sentence_classification, max_obj=data.max_obj)

        if nbest and not data.sentence_classification:
            scores, nbest_tag_seq = model.decode_nbest(batch_word, batch_features, batch_wordlen, batch_char,
                                                       batch_charlen, batch_charrecover, mask, nbest)
            nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
            nbest_pred_results += nbest_pred_result
            pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()
            ## select the best sequence to evalurate
            tag_seq = nbest_tag_seq[:, :, 0]
        else:
            tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, \
                            batch_charrecover, mask, batch_vfeature, v_modal_label, t_modal_label, v_mask, alpha=0)
        # print("tag:",tag_seq)
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover,
                                               data.sentence_classification)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances) / decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    if nbest and not data.sentence_classification:
        return speed, acc, p, r, f, nbest_pred_results, pred_scores
    return speed, acc, p, r, f, pred_results, pred_scores


def batchify_with_label(input_batch_list, gpu, if_train=False, sentence_classification=False, max_obj=3):
    if sentence_classification:
        return batchify_sentence_classification_with_label(input_batch_list, gpu, if_train)
    else:
        return batchify_sequence_labeling_with_label(input_batch_list, gpu, if_train, max_obj)


def batchify_sequence_labeling_with_label(input_batch_list, gpu, if_train=False, max_obj=3):
    """
        input: list of words, chars and labels, various length. [[words, features, chars, labels, vfeatures],[words, features, chars,labels,vfeatures],...]
            words: word ids for one sentence. (batch_size, sent_len)
            features: features ids for one sentence. (batch_size, sent_len, feature_num)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            labels: label ids for one sentence. (batch_size, sent_len)
            vfeatures: visual features for one sentence, (batch_size, 1024*n)

        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            feature_seq_tensors: [(batch_size, max_sent_len),...] list of Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
            vfeature_tensor: (batch_size, 1024)
    """
    batch_size = len(input_batch_list)

    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]
    feature_num = len(features[0][0])
    chars = [sent[2] for sent in input_batch_list]
    labels = [sent[3] for sent in input_batch_list]
    vfeatures = [sent[4] for sent in input_batch_list]

    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    vfeature_tensor = torch.zeros((batch_size, 1024 * max_obj), requires_grad=if_train).float()
    obj_lengths = torch.LongTensor([obj_f_ / 1024 for obj_f_ in list(map(len, vfeatures))])
    v_mask = torch.zeros((batch_size, max_obj), requires_grad=if_train).bool()

    for idx, (obj_feature, obj_num) in enumerate(zip(vfeatures, obj_lengths)):
        obj_num = obj_num.item()
        if obj_num > max_obj:
            obj_num = max_obj
        vfeature_tensor[idx, :1024*obj_num] = torch.Tensor(obj_feature[:1024*obj_num]).float()
        v_mask[idx, :obj_num] = torch.Tensor([1] * obj_num)

    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long())
    mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).bool()
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx, :seqlen] = torch.LongTensor(features[idx][:, idy])

    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    vfeature_tensor = vfeature_tensor[word_perm_idx]
    v_mask = v_mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad=if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)

    v_modal_label = torch.zeros((batch_size * max_obj)).long()
    t_modal_label = torch.ones((batch_size * max_seq_len)).long()
    # Pad modal label with -1
    v_pad_mask = (1 - v_mask.int()).bool().view(-1)
    v_modal_label = v_modal_label.masked_fill(v_pad_mask, -1)  # pad position should with zero attion score

    t_pad_mask = (1 - mask.int()).bool().view(-1)
    t_modal_label = t_modal_label.masked_fill(t_pad_mask, -1)

    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        vfeature_tensor = vfeature_tensor.cuda()
        v_modal_label = v_modal_label.cuda()
        t_modal_label = t_modal_label.cuda()
        mask = mask.cuda()
        v_mask = v_mask.cuda()

    return word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask, vfeature_tensor, v_modal_label, t_modal_label, v_mask


def batchify_sentence_classification_with_label(input_batch_list, gpu, if_train=True):
    """
        input: list of words, chars and labels, various length. [[words, features, chars, labels],[words, features, chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            features: features ids for one sentence. (batch_size, feature_num), each sentence has one set of feature
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            labels: label ids for one sentence. (batch_size,), each sentence has one set of feature

        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            feature_seq_tensors: [(batch_size,), ... ] list of Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, )
            mask: (batch_size, max_sent_len)
    """

    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]
    feature_num = len(features[0])
    chars = [sent[2] for sent in input_batch_list]
    labels = [sent[3] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    label_seq_tensor = torch.zeros((batch_size,), requires_grad=if_train).long()
    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long())
    mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).byte()
    label_seq_tensor = torch.LongTensor(labels)
    # exit(0)
    for idx, (seq, seqlen) in enumerate(zip(words, word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx, :seqlen] = torch.LongTensor(features[idx][:, idy])
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad=if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask


# def modal_clf_acc(t_clf_out, v_clf_out, t_modal_label, v_modal_label):
#     t_clf_pred = torch.argmax(t_clf_out, dim=1)
#     t_correct = torch.sum(t_clf_pred == t_modal_label).item()
#     t_total = torch.sum(t_modal_label == 1).item()

#     v_clf_pred = torch.argmax(v_clf_out, dim=1)
#     v_correct = torch.sum(v_clf_pred == v_modal_label).item()
#     v_total = torch.sum(v_modal_label == 0).item()

#     t_acc, v_acc, all_acc = -1, -1, -1
#     if t_total != 0:
#         t_acc = t_correct / t_total

#     if v_total != 0:
#         v_acc = v_correct / v_total

#     if t_total + v_total != 0:
#        all_acc = (t_correct + v_correct) / (t_total + v_total)

#     return t_acc, v_acc, all_acc


def train(data):
    print("Training model...")
    data.show_data_summary()
    save_data_name = data.model_dir + ".dset"
    data.save(save_data_name)

    if data.sentence_classification:
        model = SentClassifier(data)
    else:
        model = getattr(models, data.model_name)(data)

    if data.HP_gpu:
        model = model.cuda()

    # 保存初始化模型
    torch.save(model.state_dict(), data.model_dir + ".-1.model")

    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    else:
        print("Optimizer illegal: %s" % (data.optimizer))
        exit(1)
    frozen_cnt = 0
    best_dev = -10
    best_index_p = -10
    best_index_r = -10
    best_epoch = -1
    epoch_acc = []
    history_loss = []
    # data.HP_iteration = 1
    ## start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx, data.HP_iteration))
        if data.optimizer == "SGD":
            optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        sample_id = 0
        sample_loss = 0
        ner_sample_loss, v_sample_loss, t_sample_loss = 0, 0, 0
        epoch_total_loss, epoch_ner_loss, epoch_v_loss, epoch_t_loss = [], [], [], []
        # batch_acc = []
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)
        print("Shuffle: first input word list:", data.train_Ids[0][0])
        ## set model in train model
        model.train()
        model.zero_grad()
        batch_size = data.HP_batch_size
        batch_id = 0
        train_num = len(data.train_Ids)
        total_batch = train_num // batch_size + 1
        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue

            p = float(batch_id + idx * total_batch) / data.HP_iteration / total_batch
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask, batch_vfeature, v_modal_label, t_modal_label, v_mask = batchify_with_label(
                instance, data.HP_gpu, False, data.sentence_classification, max_obj=data.max_obj)

            # Train modal clf frozen
            if data.frozen_step != -1:
                frozen_cnt += 1
                if frozen_cnt >= data.frozen_step:
                    # 冻结 模态分类器层的参数
                    for param in model.modal_clf.parameters():
                        param.requires_grad = False
                    frozen_cnt = 0
                else:
                    for param in model.modal_clf.parameters():
                        param.requires_grad = True

            instance_count += 1
            loss, ner_loss, t_loss, v_loss, tag_seq, t_clf_out, v_clf_out = model.calculate_loss(batch_word,
                                                                                                 batch_features,
                                                                                                 batch_wordlen,
                                                                                                 batch_char,
                                                                                                 batch_charlen,
                                                                                                 batch_charrecover,
                                                                                                 batch_label, mask,
                                                                                                 batch_vfeature,
                                                                                                 v_modal_label,
                                                                                                 t_modal_label,
                                                                                                 v_mask, alpha)
            right, whole = predict_check(tag_seq, batch_label, mask, data.sentence_classification)
            # if t_clf_out is not None:
            #     t_acc, v_acc, all_modal_acc = modal_clf_acc(t_clf_out, v_clf_out, t_modal_label, v_modal_label)
            # else:
            #     t_acc, v_acc, all_modal_acc = 0.0, 0.0, 0.0
            right_token += right
            whole_token += whole
            # batch_acc.append((t_acc, v_acc, all_modal_acc))
            # print("loss:",loss.item())
            sample_loss += loss.item()
            # ner_sample_loss += ner_loss.item()
            # v_sample_loss += v_loss.item()
            # t_sample_loss += t_loss.item()
            # total_loss += loss.item()
            # save loss
            epoch_total_loss.append(loss.item())
            # epoch_ner_loss.append(ner_loss.item())
            # epoch_v_loss.append(v_loss.item())
            # epoch_t_loss.append(t_loss.item())

            if end % 500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (end, temp_cost, sample_loss, right_token,  whole_token, (right_token + 0.) / whole_token))
               
                if sample_loss > 1e8 or str(sample_loss) == "nan":
                    print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                    exit(1)
                sys.stdout.flush()
                sample_loss = 0
                # ner_sample_loss, v_sample_loss, t_sample_loss = 0, 0, 0

            loss.backward()
            optimizer.step()
            model.zero_grad()
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
        end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
        idx, epoch_cost, train_num / epoch_cost, total_loss))
        print("totalloss:", total_loss)
        # epoch_acc.append(batch_acc)
        # batch_acc = []
        if total_loss > 1e8 or str(total_loss) == "nan":
            print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
            exit(1)
        # continue
        speed, acc, p, r, f, _, _ = evaluate(data, model, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish
        if data.seg:
            print("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
            dev_cost, speed, acc, p, r, f))
        else:
            print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f" % (dev_cost, speed, acc))

        gc.collect()

    #     # save epoch loss
    #     history_loss.append([np.mean(epoch_total_loss), np.mean(epoch_ner_loss),
    #                          np.mean(epoch_v_loss), np.mean(epoch_t_loss)])

    # # save acc to txt
    # with open(data.log_root + '/acc.txt', 'w', encoding='utf-8') as f:
    #     # print(len(epoch_acc))
    #     f.write('t_acc,v_acc,total_acc\n')
    #     for epoch_id in range(0, len(epoch_acc)):
    #         for t, v, total in epoch_acc[epoch_id]:
    #             f.write('{},{},{}\n'.format(t, v, total))
    #         f.write('\n')

    # # save loss to txt
    # with open(data.log_root + '/loss.txt', 'w', encoding='utf-8') as f:
    #     f.write('loss,ner_loss,v_loss,t_loss\n')
    #     for ls, ner, v, t in history_loss:
    #         f.write('{},{},{},{}\n'.format(ls, ner, v, t))


def load_model_decode(data, name):
    print("Load Model from file: ", data.model_dir)
    if data.sentence_classification:
        model = SentClassifier(data)
    else:
        model = getattr(models, data.model_name)(data)

    device = 'cpu'
    if data.HP_gpu:
        model = model.cuda()
        device = 'cuda'

    model.load_state_dict(torch.load(data.load_model_dir, map_location=torch.device(device)))


    print("Decode %s data, nbest: %s ..." % (name, data.nbest))
    start_time = time.time()
    speed, acc, p, r, f, pred_results, pred_scores = evaluate(data, model, name, data.nbest)
    end_time = time.time()
    time_cost = end_time - start_time
    if data.seg:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
        name, time_cost, speed, acc, p, r, f))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f" % (name, time_cost, speed, acc))
    return pred_results, pred_scores


if __name__ == '__main__':
    train_time = time.time()
    parser = argparse.ArgumentParser(description='Tuning with NCRF++')
    # parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--config', help='Configuration File', default='None')
    parser.add_argument('--wordemb', help='Embedding for words', default='None')
    parser.add_argument('--charemb', help='Embedding for chars', default='None')
    parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--savemodel', default="data/model/saved_model.lstmcrf.")
    parser.add_argument('--savedset', help='Dir of saved data setting')
    parser.add_argument('--train', default="data/conll03/train.bmes")
    parser.add_argument('--dev', default="data/conll03/dev.bmes")
    parser.add_argument('--test', default="data/conll03/test.bmes")
    parser.add_argument('--seg', default="True")
    parser.add_argument('--raw')
    parser.add_argument('--loadmodel')
    parser.add_argument('--output')
    parser.add_argument("--seed", default="42", type=int)

    args = parser.parse_args()

    seed_num = args.seed
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)

    data = Data()
    data.HP_gpu = torch.cuda.is_available()
    if args.config == 'None':
        data.train_dir = args.train
        data.dev_dir = args.dev
        data.test_dir = args.test
        data.model_dir = args.savemodel
        data.dset_dir = args.savedset
        print("Save dset directory:", data.dset_dir)
        save_model_dir = args.savemodel
        data.word_emb_dir = args.wordemb
        data.char_emb_dir = args.charemb
        if args.seg.lower() == 'true':
            data.seg = True
        else:
            data.seg = False
        print("Seed num:", seed_num)
    else:
        data.read_config(args.config)

    # data.show_data_summary()
    status = data.status.lower()
    print("Seed num:", seed_num)
    if not os.path.exists(data.log_root):
        os.mkdir(data.log_root)

    if status == 'train':
        print("MODEL: train")
        data_initialization(data)
        data.build_visual_features()
        print('ImgID_to_feature size:', sys.getsizeof(data.imgId_to_vfeature))
        data.generate_instance('train')
        data.generate_instance('dev')
        data.generate_instance('test')
        data.build_pretrain_emb()
        train(data)
    elif status == 'decode':
        print("MODEL: decode")
        data.load(data.dset_dir)
        data.read_config(args.config)
        print(data.raw_dir)
        # exit(0)
        data.show_data_summary()
        data.generate_instance('raw')
        print("nbest: %s" % (data.nbest))
        decode_results, pred_scores = load_model_decode(data, 'raw')
        if data.nbest and not data.sentence_classification:
            data.write_nbest_decoded_results(decode_results, pred_scores, 'raw')
        else:
            data.write_decoded_results(decode_results, 'raw')
    else:
        print("Invalid argument! Please use valid arguments! (train/test/decode)")

    print('Train or Test End, Cost: {}s'.format(time.time() - train_time))
