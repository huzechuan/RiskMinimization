import time
import sys
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from collections import Counter
from random import random
import copy
from configparser import ConfigParser


class configparser(ConfigParser):
    def __init__(self,defaults=None):
        ConfigParser.__init__(self,defaults=None)

    def optionxform(self, optionstr):
        return optionstr


def log_info(info, dynamic=False):
    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if dynamic:
        print(f'\r{now_time} '
              f'{info} ', end='')
    else:
        print(f'{now_time} {info} ')
    sys.stdout.flush()

def revlut(lut):
    return {v: k for k, v in lut.items()}

def init_embedding(embedding):
    bias = np.sqrt(3.0 / embedding.size(0))
    nn.init.uniform_(embedding, -bias, bias)
    return embedding

def init_embeddings(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)


def init_lstm(input_lstm):
    """
    Initialize lstm
    """
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()

def preprocess(x):
    return None


def bert_pad(minibatch, include_length=True):
    padded = pad_sequence(minibatch, batch_first=True)
    max_len = padded.size(1)
    lengths = []
    mask = []

    if include_length:
        for x in minibatch:
            length = x.size(0)
            lengths.append(length)
            mask.append([1] * length + [0] * (max_len - length))

        mask = torch.BoolTensor(mask)#.transpose(0, 1)
        return padded, lengths, mask

    return padded

def batch_index_select(input, dim, index):
    # for ii in range(1, len(index.shape)):
    #     if ii != dim:
    #         index = index.unsqueeze(ii)
    # expanse = list(input.shape)
    # expanse[0] = -1
    # expanse[dim] = -1
    # index = index.expand(expanse)
    index_sizes = list(index.size())
    index_sizes.append(input.size(dim-1))
    index = index.view(-1)

    return torch.index_select(input, dim, index).view(index_sizes)

def to_device():

    return None


def build_noisy_label(num_candidates, datasets):
    for dataset in datasets:
        examples = dataset.examples
        for example in examples:
            candidate_labels = []
            for i in range(num_candidates):
                exec(f'candidate_labels.append(example.label{i+1})')
            candidate_labels = np.array(candidate_labels)
            label_set = candidate_labels.transpose((1, 0)).tolist()
            noisy_label = []
            for cadidates in label_set:
                tmp = Counter(cadidates)
                top_one = tmp.most_common(1)
                if top_one[0][1] == 1:  # randomly select
                    tmp_label = np.random.choice(cadidates)
                else:
                    tmp_label = top_one[0][0]
                noisy_label.append(tmp_label)
            example.noisy_label = noisy_label

def build_noisy_label_cat(num_candidates, dataset):
    # examples = dataset.examples
    examples = [copy.deepcopy(dataset.examples) for _ in range(num_candidates)]
    exampless = []
    for idx, example in enumerate(examples):
        # for i in range(num_candidates):
        for ex in example:
            noisy_label = []
            exec(f'noisy_label.append(ex.label{idx + 1})')
            ex.noisy_label = noisy_label[0]
            exampless.append(ex)
    dataset.examples = exampless


class decay_interpolation(object):
    def __init__(self, total_batches):
        super(decay_interpolation, self).__init__()
        self.epoch = 0
        self.total_batches = total_batches
        self._p = 0

    def step(self):
        self.epoch += 1
        self._p = self.epoch / self.total_batches

    def p(self):
        return self._p
