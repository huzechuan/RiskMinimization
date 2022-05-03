"""
Author:: Zechuan Hu

"""

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from processing.embeddings import TransfomerFeatures as tf
import models
import numpy as np
from models.crf import (
    CRF,
    CRF_TRI,
    CRFLoss_vb,
    CRF_SECLoss_vb
)
import models.utils as utils
from typing import Tuple, List, Dict
import copy

# import Tri_Linear.models.utils as utils
class StackedLayer(nn.Module):
    """Conditional Random Field (CRF) layer. This version is used in Lample et al. 2016, has less parameters than CRF_L.

    args:
        hidden_dim: input dim size
        tagset_size: target_set_size
        if_biase: whether allow bias in linear trans

    """

    def __init__(self, model, num_labels, if_bias=True, dropout=0.1, device='cpu', bilstm_model=False, fintune=True, **kwargs):
        super(StackedLayer, self).__init__()
        self.model = tf(model=model, device=device, fine_tune=fintune)
        self.hidden_dim = self.model.embedding_length
        # parsing tasks use static embeddings and BiLSTM.

        self.num_labels = num_labels

        self.linear_layer = nn.Linear(self.hidden_dim, self.num_labels, bias=if_bias)
        self.dropout = nn.Dropout(p=dropout)

        self._init_weight()

    def _init_weight(self):
        self.linear_layer.weight.data.normal_(mean=0.0, std=0.02)  # 0.02 for xlm-roberta
        self.linear_layer.bias.data.zero_()

    def rand_init(self):
        utils.init_linear(self.linear_layer)

    def set_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        tmp1, tmp2 = sentence
        self.seq_length = tmp2
        self.batch_size = tmp1

    def crit(self, scores, labels, masks):
        return None

    def forward(self, sents, lengths=None):
        """
        args:
            feats (batch_size, seq_len, hidden_dim) : input score from previous layers
        return:
            output from crf layer ( (batch_size * seq_len), tag_size, tag_size)
        """
        # input_ids, mask, first_subtokens = feats
        # feats = self.embeds.embed(input_ids, mask, first_subtokens)
        feats = self.model(sents)

        feats = self.dropout(feats)
        scores = self.linear_layer(feats)
        self.set_batch_seq_size((scores.size(0), scores.size(1)))

        return scores, None

def init_lstm(input_lstm):
    """
    Initialize lstm
    author:: Liyuan Liu
    """
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform_(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform_(weight, -bias, bias)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


class Softmax(StackedLayer):
    def __init__(self, model, num_labels, if_bias=True, dropout=0.1, device='cpu', **kwargs):
        super(Softmax, self).__init__(model, num_labels, if_bias, dropout, device, **kwargs)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def crit(self, scores, labels, masks):
        labels = labels.view(-1, )
        scores = scores.contiguous().view(-1, self.num_labels)
        mask_score = masks.contiguous().view(-1, 1)
        scores = scores.masked_select(mask_score.expand(-1, self.num_labels)).view(-1, self.num_labels)
        masks = masks.contiguous().view(-1, )
        labels = labels.masked_select(masks)
        loss = self.criterion(scores, labels)
        loss = loss / self.batch_size # 5e-5
        return loss

    def decode(self, scores, masks=None, view=None, confidence=False):
        if isinstance(scores, tuple):
            probs, tags = torch.max(scores[0], 2)
        else:
            probs, tags = torch.max(scores, 2)
        if confidence:
            return probs, tags
        return tags

def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class KD(Softmax):
    def __init__(self, model, num_labels, if_bias=True, dropout=0.1, device='cpu', num_source=None, interpolation=1, **kwargs):
        super(KD, self).__init__(model, num_labels, if_bias, dropout, device, **kwargs)
        net = Softmax(model=model, num_labels=self.num_labels, dropout=0.1, device=device)
        self.source_models = clone(net, num_source)
        self.num_source = num_source
        self.source_softmax = nn.Softmax(dim=-1)
        self.target_softmax = nn.LogSoftmax(dim=-1)
        self.kld = nn.KLDivLoss(reduction='sum')
        self.device = device

        self.interpolation = interpolation
        self.criterion = nn.NLLLoss(reduction='sum')

    def load_source_models(self, model_paths):
        for i, model_path in enumerate(model_paths):
            utils.log_info(f'loading source model: {model_path}')
            model_config = torch.load(model_path, map_location=self.device)
            self.source_models[i].load_state_dict(model_config['model_state_dict'])
            self.source_models[i].model.fine_tune = False

    def crit(self, p, q, mask, labels=None):
        # p: target log probability
        nums = p.size(-1)
        num_instance = p.size(0)
        p = p.contiguous().view(-1, nums)
        q = q.contiguous().view(-1, nums)

        mask = mask.contiguous().view(-1, 1)
        p = p.masked_select(mask.expand_as(p)).view(-1, nums)

        q = q.masked_select(mask.expand_as(q)).view(-1, nums)

        if labels is not None:
            labels = labels.view(-1, )
            # scores = scores.contiguous().view(-1, self.num_labels)
            # mask_score = masks.contiguous().view(-1, 1)
            # scores = scores.masked_select(mask_score.expand(-1, self.num_labels)).view(-1, self.num_labels)
            mask = mask.contiguous().view(-1, )
            labels = labels.masked_select(mask)
            labeled_loss = self.criterion(p, labels)
            return self.kld(p, q) + self.interpolation * labeled_loss
        return self.kld(p, q)

    def forward(self, sents, lengths=None):
        source_prob = []
        with torch.no_grad():
            for source_model in self.source_models:
                source_model.eval()
                source_prob.append(self.source_softmax(source_model(sents, lengths)[0]))
        source_prob = torch.stack(source_prob, dim=-1)
        source_prob = torch.mean(source_prob, dim=-1)

        feats = self.model(sents)
        feats = self.dropout(feats)
        scores = self.linear_layer(feats)
        self.set_batch_seq_size((scores.size(0), scores.size(1)))
        target_log_prob = self.target_softmax(scores)

        return target_log_prob, source_prob

    def decode(self, scores, masks=None, view=None, confidence=False):
        if isinstance(scores, tuple):
            probs, tags = torch.max(scores[0], 2)
        else:
            probs, tags = torch.max(scores, 2)
        if confidence:
            return probs, tags
        return tags

class Latent_KD(KD):
    def __init__(self, model, num_labels, if_bias=True, dropout=0.1, device='cpu', num_source=None, mu=3, interpolation=1, **kwargs):
        super(Latent_KD, self).__init__(model, num_labels, if_bias, dropout, device, num_source, **kwargs)
        # net = Softmax(model=model, num_labels=self.num_labels, dropout=0.1, device=device)
        # self.source_models = clone(net, num_source)
        self.num_source = num_source
        self.source_softmax = nn.Softmax(dim=-1)
        self.target_softmax = nn.LogSoftmax(dim=-1)
        self.kld = nn.KLDivLoss(reduction='sum')
        self.device = device

        self.mu = mu
        self.mapping_mat = nn.Parameter(self.mu * torch.eye(self.num_labels, dtype=torch.float32))
        self.normalize_map = nn.LogSoftmax(dim=0)

        self.interpolation = interpolation

    def load_source_models(self, model_paths):
        for i, model_path in enumerate(model_paths):
            utils.log_info(f'loading source model: {model_path}')
            model_config = torch.load(model_path, map_location=self.device)
            self.source_models[i].load_state_dict(model_config['model_state_dict'])
            self.source_models[i].model.fine_tune = False

    def forward(self, sents, lengths=None):
        source_prob, source_predictions = [], None
        with torch.no_grad():
            for source_model in self.source_models:
                source_model.eval()
                source_scores = source_model(sents, lengths)[0]
                source_prob.append(self.source_softmax(source_scores))
                source_predictions = self.decode(source_scores) # TODO
        normalized_mat = self.normalize_map(self.mapping_mat)
        risks = utils.batch_index_select(normalized_mat, 0, source_predictions)
        source_prob = torch.stack(source_prob, dim=-1)
        source_prob = torch.mean(source_prob, dim=-1)

        feats = self.model(sents)
        feats = self.dropout(feats)
        scores = self.linear_layer(feats)
        self.set_batch_seq_size((scores.size(0), scores.size(1)))
        target_log_prob = self.target_softmax(scores)
        target_log_prob = risks + target_log_prob

        return target_log_prob, source_prob


class LSTM_Net(nn.Module):
    """LSTM_CRF model

    args:
        vocab_size: size of word dictionary
        num_labels: size of label set
        embedding_dim: size of word embedding
        hidden_dim: size of word-level blstm hidden dim
        rnn_layers: number of word-level lstm layers
        dropout_ratio: dropout ratio
        large_CRF: use CRF_L or not, refer model.crf.CRF_L and model.crf.CRF_S for more details
    """

    def __init__(self, tag_map: Dict, embedding_length: int,
                 hidden_dim: int, rnn_layers: int,
                 dropout_ratio: float, use_crf=True):
        super(LSTM_Net, self).__init__()
        self.embedding_dim: int = embedding_length
        self.hidden_dim: int = hidden_dim
        self.use_crf: bool = use_crf
        self.tag_map: Dict = tag_map

        # RNN Module
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim // 2,
                            num_layers=rnn_layers, bidirectional=True)
        self.rnn_layers: int = rnn_layers

        # Dropout Module
        self.dropout1 = nn.Dropout(p=dropout_ratio)
        self.dropout2 = nn.Dropout(p=dropout_ratio)

        self.num_labels: int = len(tag_map)
        if use_crf:
            self.top_layer = CRF(hidden_dim, self.num_labels)
            self.loss = CRFLoss_vb(self.num_labels, self.tag_map['<start>'], self.tag_map['<pad>'])
        else:
            self.top_layer = nn.Linear(hidden_dim, self.num_labels, bias=True)
            # CrossEntropyLoss
            self.loss = nn.CrossEntropyLoss(reduction='sum')

        self.batch_size: int = None
        self.seq_length: int = None

    def set_batch_size(self, bsize):
        """
        set batch size
        """
        self.batch_size = bsize

    def set_batch_seq_size(self, sizes: Tuple):
        """
        set batch size and sequence length
        """
        if isinstance(sizes, Tuple):
            self.batch_size, self.seq_length = sizes
        else:
            self.batch_size, self.seq_length = sizes.size(1), sizes.size(0)

    def rand_init(self):
        """
        random initialization

        args:
            init_embedding: random initialize embedding or not
        """
        init_lstm(self.lstm)
        if self.use_crf:
            self.top_layer.rand_init()

    def crit(self, scores, tags, masks):
        if self.use_crf:
            loss = self.loss(scores, tags, masks)
        else:
            # Sentences averaged version
            tags = tags.view(-1, )
            scores = scores.view(-1, self.num_labels)
            scores = scores.masked_select(masks.view(-1, 1).expand(-1, self.num_labels)).view(-1, self.num_labels)
            masks = masks.view(-1, )
            tags = tags.masked_select(masks)
            loss = self.loss(scores, tags)
            loss = loss / self.batch_size

        return loss

    def forward(self, feats, lengths, hidden=None):
        '''
        args:
            sentence (word_seq_len, batch_size) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf output (word_seq_len, batch_size, tag_size, tag_size), hidden
        '''
        # embeds = self.embeddings.embed(feats)
        embeds = feats
        self.set_batch_seq_size(embeds)
        total_len = self.seq_length#feats[0].total_len

        d_embeds = self.dropout1(embeds)

        d_embeds = pack_padded_sequence(input=d_embeds, lengths=lengths, batch_first=False, enforce_sorted=False)
        lstm_out, hidden = self.lstm(d_embeds, hidden)
        # lstm_out = lstm_out.view(-1, self.hidden_dim)
        lstm_out, batch_lens = pad_packed_sequence(lstm_out, batch_first=False, padding_value=0.0, total_length=total_len)

        d_lstm_out = self.dropout2(lstm_out)

        score = self.top_layer(d_lstm_out)

        if self.use_crf:
            score = score.view(self.seq_length, self.batch_size, self.num_labels, self.num_labels)
        else:
            score = score.view(self.seq_length, self.batch_size, self.num_labels)

        return score, hidden

    def decode(self, scores):
        _, tags = torch.max(scores, 2)

        return tags

class LSTM_Model(nn.Module):
    """LSTM_CRF model

    args:
        vocab_size: size of word dictionary
        num_labels: size of label set
        embedding_dim: size of word embedding
        hidden_dim: size of word-level blstm hidden dim
        rnn_layers: number of word-level lstm layers
        dropout_ratio: dropout ratio
        large_CRF: use CRF_L or not, refer model.crf.CRF_L and model.crf.CRF_S for more details
    """

    def __init__(self, tag_map, embeddings, hidden_dim, rnn_layers, dropout_ratio,
                 use_crf=True, tri_parameter=None):
        super(LSTM_Model, self).__init__()
        self.embeddings = embeddings
        self.embedding_dim = embeddings.embedding_length
        self.hidden_dim = hidden_dim
        self.use_crf = use_crf
        self.tag_map = tag_map
        # self.vocab_size = vocab_size
        #
        # self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim // 2,
                            num_layers=rnn_layers, bidirectional=True)
        self.rnn_layers = rnn_layers

        self.dropout1 = nn.Dropout(p=dropout_ratio)
        self.dropout2 = nn.Dropout(p=dropout_ratio)

        self.num_labels = len(tag_map)
        if use_crf:
            self.top_layer = CRF(hidden_dim, self.num_labels, tri_parameter)
            self.loss = CRFLoss_vb(self.num_labels, self.tag_map['<start>'], self.tag_map['<pad>'])
            # self.loss = CRF_SECLoss_vb(self.num_labels, self.tag_map['<START>'], self.tag_map['<PAD>'])
        else:
            self.top_layer = nn.Linear(hidden_dim, self.num_labels, bias=True)
            self.loss = nn.CrossEntropyLoss(reduction='sum')

        self.batch_size = None
        self.seq_length = None

    def set_batch_size(self, bsize):
        """
        set batch size
        """
        self.batch_size = bsize

    def set_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        tmp = sentence.size()
        self.seq_length = tmp[0]
        self.batch_size = tmp[1]


    def rand_init(self, init_embedding=False):
        """
        random initialization

        args:
            init_embedding: random initialize embedding or not
        """

        init_lstm(self.lstm)
        if self.use_crf:
            self.top_layer.rand_init()

    def load_pretrained_embedding(self, pre_embeddings):
        """
        load pre-trained word embedding

        args:
            pre_word_embeddings (self.word_size, self.word_dim) : pre-trained embedding
        """
        assert (pre_embeddings.size()[1] == self.embedding_dim)
        self.word_embeds.weight = nn.Parameter(pre_embeddings)

    def crit(self, scores, tags, masks):
        if self.use_crf:
            loss = self.loss(scores, tags, masks)
        else:
            # Sentences averaged version
            tags = tags.view(-1, )
            scores = scores.view(-1, self.num_labels)
            scores = scores.masked_select(masks.view(-1, 1).expand(-1, self.num_labels)).view(-1, self.num_labels)
            masks = masks.view(-1, )
            tags = tags.masked_select(masks)
            loss = self.loss(scores, tags)
            loss = loss / self.batch_size

        # else:
        #     # Token averaged version
        #     loss = 0
        #     scores = scores.transpose(0, 1)
        #     tags = tags.transpose(0, 1)
        #     masks = masks.transpose(0, 1)
        #     for score, tag, mask in zip(scores, tags, masks):
        #         tag = tag.view(-1, )
        #         score = score.view(-1, self.num_labels)
        #         score = score.masked_select(mask.view(-1, 1).expand(-1, self.num_labels)).view(-1, self.num_labels)
        #         mask = mask.view(-1, )
        #         tag = tag.masked_select(mask)
        #         loss += self.loss(score, tag)

        return loss #/ self.batch_size

    def forward(self, feats, hidden=None):
        '''
        args:
            sentence (word_seq_len, batch_size) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf output (word_seq_len, batch_size, tag_size, tag_size), hidden
        '''
        embeds = self.embeddings.embed(feats)
        self.set_batch_seq_size(embeds)
        total_len = self.seq_length#feats[0].total_len
        true_len = feats.true_len

        lengths = feats.word_lengths#torch.IntTensor([sent.length for sent in feats])
        d_embeds = self.dropout1(embeds)

        d_embeds = pack_padded_sequence(input=d_embeds, lengths=lengths, batch_first=False, enforce_sorted=False)
        lstm_out, hidden = self.lstm(d_embeds, hidden)
        # lstm_out = lstm_out.view(-1, self.hidden_dim)
        lstm_out, batch_lens = pad_packed_sequence(lstm_out, batch_first=False, padding_value=0.0, total_length=total_len)

        d_lstm_out = self.dropout2(lstm_out)

        score = self.top_layer(d_lstm_out)

        if self.use_crf:
            score = score.view(true_len, self.batch_size, self.num_labels, self.num_labels)
        else:
            score = score.view(true_len, self.batch_size, self.num_labels)

        return score, hidden


class LSTM_SEC_CRF(nn.Module):
    """LSTM_SEC_ORDER_CRF model
    author: huzechuan@std.uestc.edu.cn
    args:
        vocab_size: size of word dictionary
        num_labels: size of label set
        embedding_dim: size of word embedding
        hidden_dim: size of word-level blstm hidden dim
        rnn_layers: number of word-level lstm layers
        dropout_ratio: dropout ratio
    """

    def __init__(self, vocab_size, num_labels, embedding_dim, hidden_dim, rnn_layers, dropout_ratio):
        super(LSTM_SEC_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=rnn_layers, bidirectional=True)
        self.rnn_layers = rnn_layers

        self.dropout1 = nn.Dropout(p=dropout_ratio)
        self.dropout2 = nn.Dropout(p=dropout_ratio)

        self.num_labels = num_labels

        self.crf = crf.CRF_S_SEC(hidden_dim, num_labels)

        self.batch_size = 1
        self.seq_length = 1

    def rand_init_hidden(self):
        """
        random initialize hidden variable
        """
        return autograd.Variable(
            torch.randn(2 * self.rnn_layers, self.batch_size, self.hidden_dim // 2)), autograd.Variable(
            torch.randn(2 * self.rnn_layers, self.batch_size, self.hidden_dim // 2))

    def set_batch_size(self, bsize):
        """
        set batch size
        """
        self.batch_size = bsize

    def set_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        tmp = sentence.size()
        self.seq_length = tmp[0]
        self.batch_size = tmp[1]

    def load_pretrained_embedding(self, pre_embeddings):
        """
        load pre-trained word embedding

        args:
            pre_word_embeddings (self.word_size, self.word_dim) : pre-trained embedding
        """
        assert (pre_embeddings.size()[1] == self.embedding_dim)
        self.word_embeds.weight = nn.Parameter(pre_embeddings)



    def rand_init(self, init_embedding=False):
        """
        random initialization

        args:
            init_embedding: random initialize embedding or not
        """
        utils.init_lstm(self.lstm)
        self.crf.rand_init()

    def forward(self, sentence, hidden=None):
        '''
        args:
            sentence (word_seq_len, batch_size) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf output (word_seq_len, batch_size, tag_size, tag_size), hidden
        '''
        self.set_batch_seq_size(sentence)

        embeds = self.word_embeds(sentence)
        d_embeds = self.dropout1(embeds)

        lstm_out, hidden = self.lstm(d_embeds, hidden)
        lstm_out = lstm_out.view(-1, self.hidden_dim)

        d_lstm_out = self.dropout2(lstm_out)

        crf_out = self.crf(d_lstm_out)
        crf_out = crf_out.view(self.seq_length, self.batch_size, self.num_labels, self.num_labels,
                               self.num_labels)

        return crf_out, hidden


def dismiss_relations(ent1, ent2, table):
    import itertools as it
    for e1, e2 in it.product(ent1, ent2):
        table[e1, e2] = -10000.0
        table[e2, e1] = -10000.0
    return None

def init_fix_table(table, tag_dic):
    dic = {'LOC': [], 'PER': [], 'ORG': [], 'MISC': []}
    for key, item, in tag_dic.items():
        entity = key.split('-')[-1]
        if entity in dic.keys(): dic[entity].append(item)
    for k in dic.keys():
        for ak in dic.keys():
            if k == ak: continue
            dismiss_relations(dic[k], dic[ak], table)
    table[0] = -10000.0
    table[1] = -10000.0
    table[:, 0] = -10000.0
    table[:, 1] = -10000.0
    return None

class Decoder_Net(nn.Module):
    def __init__(self, tag_map: Dict, edit_vec_len:int,
                 embedding_length: int,
                 use_crf=True, gold_table=None, mu=2, method='mrt', model=None, device=None):
        super(Decoder_Net, self).__init__()
        self.method = method
        self.num_labels = len(tag_map)
        self.model = Softmax(model=model, num_labels=self.num_labels, dropout=0.1, device=device)
        self.mu = mu
        self.edit_vec_len = edit_vec_len
        self.edit_vec = nn.Parameter(torch.FloatTensor(self.edit_vec_len))
        # self.dense = nn.Linear(self.edit_vec_len, self.num_labels)
        self.label_embeddings = nn.Parameter(self.mu*torch.eye(self.num_labels, dtype=torch.float32))
        # self.label_embeddings = nn.Parameter(torch.eye(self.num_labels, dtype=torch.float32))
        # self.label_embeddings = nn.Parameter(0.5*torch.eye(self.num_labels, dtype=torch.float32))
        # self.label_embeddings = nn.Parameter(20 * torch.eye(self.num_labels, dtype=torch.float32))
        # self.label_embeddings = nn.Parameter(torch.FloatTensor(self.num_labels, self.num_labels))
        # self.label_embeddings = nn.Parameter(torch.FloatTensor(gold_table))
        # self.label_embeddings = torch.FloatTensor(gold_table).cuda()
        # self.label_embeddings = None

        self.fix_table = torch.zeros(self.num_labels, self.num_labels, dtype=torch.float32).cuda()
        # init_fix_table(self.fix_table, tag_map)
        # print(self.fix_table)

        self.prob_u_func = nn.CrossEntropyLoss(reduction='none')
        self.prob_ = nn.Softmax(dim=0)
        self.source_prob = nn.Softmax(dim=-1)

        # self.soft_prob = nn.Softmax(dim=1)
        self.flag = True
        self.vector = torch.zeros([1, self.num_labels], dtype=torch.float32)
        self.epoch = 0
        self.device = device

    def set_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        tmp1, tmp2 = sentence
        self.seq_length = tmp2
        self.batch_size = tmp1

    def load_source_model(self, model_path):
        utils.log_info(f'loading source model: {model_path}')
        model_config = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(model_config['model_state_dict'])
        self.model.model.fine_tune = False

    def set_table(self, m):
        self.label_embeddings = None
        # self.label_embeddings = nn.Parameter(torch.FloatTensor(m))
        self.label_embeddings = torch.FloatTensor(m).cuda()

    def reset_label(self):
        vec = torch.min(self.label_embeddings.data, dim=0)[0]
        vec[2] = 20
        # vector = torch.zeros([1, self.num_labels], dtype=torch.float32)
        # if vec[2] < 4:
        #     vector[0, 2] = 20
        # else:
        #     vector[0, 2] =
        self.label_embeddings.data[2, :] = vec

    def rand_init(self):
        """
                random initialization

                args:
                    init_embedding: random initialize embedding or not
                """
        # utils.init_embeddings(self.label_embeddings)
        # init_lstm(self.lstm)
        # utils.init_linear(self.dense)
        # if self.use_crf:
        #     self.top_layer.rand_init()

        utils.init_embedding(self.edit_vec)

    def cat_label_y(self, features, transitions):
        # features = features.unsqueeze(2).repeat(1, 1, self.num_labels, 1)
        # label_features = self.label_embeddings.unsqueeze(0).repeat(self.batch_size, 1, 1)
        # label_features = label_features.unsqueeze(0).repeat(self.seq_length, 1, 1, 1)
        #
        # features = torch.cat([features, label_features], -1)
        transitions = transitions.view(1, self.num_labels, self.num_labels).expand(self.batch_size,
                                                                                   self.num_labels,
                                                                                   self.num_labels)
        return transitions.view(1, self.batch_size,
                                self.num_labels,
                                self.num_labels).expand(self.seq_length,
                                                        self.batch_size,
                                                        self.num_labels,
                                                        self.num_labels)
        features = features.view(self.seq_length, self.batch_size, self.num_labels, 1)
        features = features.expand(self.seq_length, self.batch_size, self.num_labels, self.num_labels)

        transitions = transitions.view(1, self.num_labels, self.num_labels).expand(self.batch_size,
                                                                                   self.num_labels,
                                                                                   self.num_labels)

        features = features + transitions.view(1, self.batch_size,
                                               self.num_labels,
                                               self.num_labels).expand(self.seq_length,
                                                                       self.batch_size,
                                                                       self.num_labels,
                                                                       self.num_labels)
        return features

    def crit(self, scores, tags, masks):
        scores = -self.prob(scores, tags)
        scores = scores[:, :, 3]
        scores = scores.masked_select(masks)
        return torch.sum(scores) / tags.size(1)

    def norm_loss(self):
        diag_val = torch.diag(self.label_embeddings) # select the main diagonal values
        l2_norm = torch.norm(diag_val)
        # indexs = torch.LongTensor(range(19)).cuda()
        # cross_norm = self.prob_u_func(self.label_embeddings, indexs)
        # return torch.sum(cross_norm)
        return l2_norm

    def prob(self, scores, labels):
        # prob = utils.batch_index_select(scores, 0, labels)
        # log_prob = torch.log(prob)
        log_prob = torch.log(scores)
        return log_prob.view(self.batch_size, self.seq_length, -1)

    def cat_edit_vec(self, features):
        edit_vec = self.edit_vec.unsqueeze(0).repeat(self.batch_size, 1)
        edit_vec = edit_vec.unsqueeze(0).repeat(self.seq_length, 1, 1)

        # features = torch.cat([features, edit_vec], -1)
        features = edit_vec

        return features.contiguous().view(self.seq_length, self.batch_size, -1)

    def forward(self, features, lengths, transition, hidden=None, p=0.):
        total_len = self.seq_length  # feats[0].total_len
        with torch.no_grad():
            logits, _ = self.model(features, lengths)
            p_s = self.source_prob(logits)
        if self.method == 'mrt':
            tr = self.prob_(transition)
        else:
            tr = self.prob_(transition) * (1 - p) + self.prob_(self.label_embeddings) * p
        # if self.epoch == epoch:
        #     self.epoch += 1
        tr = torch.matmul(p_s, tr)
        return tr
        # return self.cat_label_y(features, tr)
        # return self.cat_label_y(features, transition)
        # return self.cat_label_y(features, self.label_embeddings)
        # return self.cat_label_y(features, self.label_embeddings + self.fix_table)
        # return self.cat_label_y(features, self.label_embeddings + transition)
        # return self.soft_prob(self.label_embeddings)
        # return self.label_embeddings

        # lengths = lengths.view(self.batch_size, 1).expand(self.batch_size, self.num_labels)
        # lengths = lengths.reshape(self.batch_size * self.num_labels)
        d_embeds = self.dropout1(features)

        d_embeds = pack_padded_sequence(input=d_embeds, lengths=lengths, batch_first=False, enforce_sorted=False)
        lstm_out, hidden = self.lstm(d_embeds, hidden)
        # lstm_out = lstm_out.view(-1, self.hidden_dim)
        lstm_out, batch_lens = pad_packed_sequence(lstm_out, batch_first=False, padding_value=0.0,
                                                   total_length=total_len)

        d_lstm_out = self.dropout2(lstm_out)

        score = self.top_layer(d_lstm_out)
        score = score.view(self.seq_length, self.batch_size, self.num_labels)

        if self.use_crf:
            score = score.view(total_len, self.batch_size, self.num_labels, self.num_labels)
        else:
            score = self.cat_label_y(score, transition)
            # score = score.view(-1, self.num_labels, self.num_labels).transpose(1, 2)
            pass

        return score


class Encoder_Net(Softmax):
    def __init__(self, tag_map: Dict, embedding: str,
                 dropout_ratio: float, device=True):
        super(Softmax, self).__init__(embedding, len(tag_map), dropout=dropout_ratio, device=device)
        self.log_prob_func = nn.LogSoftmax(dim=2)

    # def set_batch_seq_size(self, score):
    #     bat, seq = score[:, :, 0].size()
    #     self.seq_length = seq
    #     self.batch_size = bat

    def prob(self, scores):

        return self.log_prob_func(scores)
    #
    # def forward(self, sents):
    #     # self.set_batch_seq_size((features.size(1), features.size(0)))
    #     scores, _ = .forward(sents)
    #     total_len = self.seq_length  # feats[0].total_len
    #     # true_len = features.true_len
    #
    #     d_hiddens = self.dropout1(scores)
    #
    #     return d_hiddens

