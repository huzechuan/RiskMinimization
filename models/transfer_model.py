import torch
import torch.nn as nn
from models.lstm_model import Encoder_Net, Decoder_Net
from typing import List, Tuple, Dict
from processing.embeddings import StackedEmbeddings
import models.utils as utils
from tools.table_statistic import summation
import copy


def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Transfer_Model(nn.Module):
    def __init__(self, tag_map: Dict, embeddings: str,
                 dropout_ratio: float,
                 model_type: str='SoftmaxX2', num_decoders: int=1,
                 corpus=None, softem=True, mu=2, device='cpu', method='mrt', init_mu=4):
        super(Transfer_Model, self).__init__()
        self.softem = softem
        self.embeddings = embeddings
        self.encoder_net = Encoder_Net(tag_map, self.embeddings,
                                       dropout_ratio, device=device)

        self.label_embed_len = 20
        self.edit_vec_len = 25
        self.num_labels = len(tag_map)
        # self.label_embeddings = nn.Parameter(torch.FloatTensor(self.num_labels, self.num_labels))
        # self.label_embeddings = nn.Parameter(20 * torch.eye(self.num_labels, dtype=torch.float32))
        self.label_embeddings = torch.eye(self.num_labels, dtype=torch.float32, requires_grad=False).cuda()
        # self.label_embeddings = nn.Parameter(2*torch.eye(self.num_labels, dtype=torch.float32))

        # self.label_embeddings *= mu
        self.label_embeddings *= init_mu
        self.decoder_nets = []
        # label_map, table_list = summation(corpus, 'ner')
        label_map, table_list = None, [None] * 7

        # self.label_embeddings = torch.FloatTensor(table_list).cuda()

        print(label_map, tag_map)
        # assert label_map == tag_map, 'Label dic not match!'
        net = Decoder_Net(tag_map, self.edit_vec_len,
                          self.edit_vec_len,
                          use_crf=False, mu=mu, method=method, model=self.embeddings, device=device)
        utils.log_info(net.method)
        self.decoder_nets = clone(net, num_decoders)
        # for i in range(num_decoders):
        #     net = Decoder_Net(tag_map, self.edit_vec_len,
        #                       self.edit_vec_len,
        #                       use_crf=False, gold_table=table_list[i])
        #     self.add_module(f'decoder_net{i}', net)
        #     self.decoder_nets.append(net)
        self.model_type = model_type

        self.batch_size = None
        self.seq_length = None
        self.num_decoders = num_decoders
        self.tag_map = tag_map

        self.q_func = nn.Softmax(dim=2)
        self.flag = True

        self.big_prob_rate = 0.0
        self.device = device

    def reset_table(self, M):
        for n, m in zip(self.decoder_nets, M):
            n.set_table(m)

    def reset_vectors(self):
        for i, decoder_net in enumerate(self.decoder_nets):
            decoder_net.reset_label()

    def rand_init(self):
        # utils.init_embeddings(self.label_embeddings)
        # self.label_embeddings.data.zero_()
        self.encoder_net.rand_init()
        for i, decoder_net in enumerate(self.decoder_nets):
            decoder_net.rand_init()

    def load_source_model(self, model_paths):
        for i, model_path in enumerate(model_paths):
            self.decoder_nets[i].load_source_model(model_path)

    def load_encoder(self, params):
        self.encoder_net.load_state_dict(params.static_model['model_state_dict'])
        print('load encoder model!')

    def set_batch_seq(self, sizes: Tuple):
        self.encoder_net.set_batch_seq_size(sizes)
        for i, decoder_net in enumerate(self.decoder_nets):
            decoder_net.set_batch_seq_size(sizes)
        self.batch_size, self.seq_length = sizes

    def cat_label_y(self, features):
        # features = features.unsqueeze(2).repeat(1, 1, self.num_labels, 1)
        # label_features = self.label_embeddings.unsqueeze(0).repeat(self.batch_size, 1, 1)
        # label_features = label_features.unsqueeze(0).repeat(self.seq_length, 1, 1, 1)
        #
        # features = torch.cat([features, label_features], -1)
        features = features.view(self.seq_length, self.batch_size, self.num_labels, 1)
        features = features.expand(self.seq_length, self.batch_size, self.num_labels, self.num_labels)

        transitions = self.label_embeddings.view(1, self.num_labels, self.num_labels).expand(self.batch_size)
        features = features + transitions.view(1, self.batch_size,
                                                         self.num_labels,
                                                         self.num_labels).expand(self.seq_length,
                                                                                 self.batch_size,
                                                                                 self.num_labels,
                                                                                 self.num_labels)
        return features

    def log_joint_prob(self, encoder_score, decoders_scores, labels):
        encoder_probs = self.encoder_net.prob(encoder_score)
        # a = torch.tensor(torch.ge(torch.max(encoder_probs, -1)[0], -0.105), dtype=torch.int32).numpy()
        # import numpy as np
        # rate = np.sum(a) / (a.shape[0] * a.shape[1])
        # if rate > 0.5:
        #     s = 3
        decoders_probs = []
        for decoder_score, decoder_net, label in zip(decoders_scores, self.decoder_nets, labels):
            decoders_probs.append(decoder_net.prob(decoder_score, label))

        values = 1
        joint_prob = 0.
        for decoder_prob in decoders_probs:
            joint_prob += decoder_prob + encoder_probs

        return joint_prob# + encoder_probs

    def q_value(self, encoder_score, decoders_scores, labels):
        joint_prob = self.log_joint_prob(encoder_score, decoders_scores, labels)

        # values = torch.exp(joint_prob)
        #
        # sum_y = torch.sum(values, dim=-1).view(self.seq_length, self.batch_size, 1)
        # q_value = torch.div(values, sum_y)
        if self.softem:
            q_value = self.q_func(joint_prob)
        else:
            joint_prob = joint_prob.view(-1, self.num_labels)
            # print(joint_prob)
            samples = joint_prob.size(0)
            ind = torch.argmax(joint_prob, dim=-1)
            q_value = torch.zeros_like(joint_prob)
            q_value[range(samples), ind] = 1
            q_value = q_value.view(self.batch_size, self.seq_length, self.num_labels)
            # print(q_value)
            if self.flag:
                print('Hard EM.')
                self.flag = False
            # exit(0)
        return q_value

    def loss(self, q_valus, encoder_score, decoders_scores, labels, mask):
        joint_prob = self.log_joint_prob(encoder_score, decoders_scores, labels)

        # loss = q_valus * joint_prob - q_valus * torch.log((q_valus))
        loss = torch.mul(q_valus, joint_prob)

        # loss = loss.masked_select(mask.view(self.seq_length, self.batch_size, 1))
        loss = torch.sum(loss, dim=2)
        loss = loss.masked_select(mask)
        loss = (-torch.sum(loss)) / self.batch_size

        if_norm = False
        if if_norm:
            lamb = 100
            decoder_norms = 0
            for decoder in self.decoder_nets:
                decoder_norms += decoder.norm_loss()
            # return loss * lamb + decoder_norms * (1 - lamb)
            return loss - decoder_norms * lamb
        return loss#, self.batch_size

    def mrk_q(self, encoder_score, decoders_scores, labels):
        encoder_probs = self.encoder_net.prob(encoder_score)
        decoders_probs = []
        for decoder_score, decoder_net, label in zip(decoders_scores, self.decoder_nets, labels):
            decoders_probs.append(decoder_net.prob(decoder_score, label))

        values = 1
        joint_prob = 0.
        for decoder_prob in decoders_probs:
            joint_prob += decoder_prob + encoder_probs
        return joint_prob

    def mrk_loss(self, encoder_score, decoders_scores, labels, mask):
        encoder_probs = self.encoder_net.prob(encoder_score)
        decoders_probs, log_joint_p = [], []
        for decoder_score, decoder_net, label in zip(decoders_scores, self.decoder_nets, labels):
            decoder_prob = decoder_net.prob(decoder_score, label)
            decoders_probs.append(decoder_prob)
            log_joint_p.append(decoder_prob + encoder_probs)
        sum_u = 0.0
        for log_p in log_joint_p:
            sum_u += torch.log(torch.sum(torch.exp(log_p), dim=-1))
        # assert list(sum_u.size()) == [self.batch_size, self.seq_length], f'Wrong size: tensor {sum_u.size()}, except {[self.seq_length, self.batch_size]}'

        sum_u = sum_u.masked_select(mask)
        return -torch.sum(sum_u) / self.batch_size

    def forward(self, features, lengths, p=0.):
        encoder_score, _ = self.encoder_net(features)
        self.set_batch_seq((encoder_score.size(0), encoder_score.size(1)))

        decoders_scores = []
        for i, decoder_net in enumerate(self.decoder_nets):
            decoder_score = decoder_net(features, lengths, self.label_embeddings, p=p)
            decoders_scores.append(decoder_score)

        return (encoder_score, decoders_scores)

    def decode(self, q_values):
        _, tags = torch.max(q_values, 2)

        return tags
    # def decode(self, encoder_score, decoders_scores, labels):
    #     log_p = self.log_joint_prob(encoder_score, decoders_scores, labels)
    #     p = torch.exp(log_p)
    #     _, tags = torch.max(encoder_score, dim=2)
    #     return tags.cpu()

    def crit(self, encoder_score, decoders_scores, labels, mask):
        # loss = self.decoder_nets[0].crit(scores, tags, masks)
        log_p = self.log_joint_prob(encoder_score, decoders_scores, labels)
        p = torch.exp(log_p)
        sum_y = torch.sum(p, dim=-1)
        loss = torch.sum(-torch.log(sum_y.masked_select(mask)))
        return loss / self.batch_size
