"""
.. module:: evaluator
    :synopsis: evaluation method (f1 score and accuracy)

.. moduleauthor:: Liyuan Liu, Frank Xu
"""

import torch
import numpy as np
import itertools

import models.utils as utils
from torch.autograd import Variable
from tqdm import tqdm
import sys
from models.crf import CRFDecode_vb
from models.crf import CRFSECDecode_vb
import os
import subprocess
import re
from pathlib import Path


def repack(batch):
    t = []
    m = []
    for b in batch:
        t.append(b.tag)
        m.append(b.mask)
    tags = torch.cat(t, 0)
    masks = torch.cat(m, 0)
    return tags, masks.transpose(0, 1).cuda()


class eval_batch:
    """Base class for evaluation, provide method to calculate f1 score and accuracy

    args:
        packer: provide method to convert target into original space [TODO: need to improve]
        l_map: dictionary for labels
    """

    def __init__(self, l_map, file_path=None):
        self.l_map = l_map
        self.r_l_map = utils.revlut(l_map)
        self.file_path = file_path

    def reset(self):
        """
        re-set all states
        """
        self.correct_labels = 0
        self.total_labels = 0

    def calc_acc_batch(self, decoded_data, target_data, *args):
        """
        update statics for accuracy

        args:
            decoded_data (batch_size, seq_len): prediction sequence
            target_data (batch_size, seq_len): ground-truth
        """
        batch_decoded = torch.unbind(decoded_data, 1)
        batch_targets = torch.unbind(torch.squeeze(target_data.cpu()).transpose(0, 1), 0)

        for decoded, target in zip(batch_decoded, batch_targets):
            gold = target % len(self.l_map)
            # remove padding
            length = utils.find_length_from_labels(gold, self.l_map)
            gold = gold[:length].numpy()
            best_path = decoded[:length].numpy()

            self.total_labels += length
            self.correct_labels += np.sum(np.equal(best_path, gold))

    def write_result(self, decoded_data, target_data, feature, fout):

        batch_decoded = torch.unbind(decoded_data, 1)
        batch_targets = torch.unbind(torch.squeeze(target_data.cpu(), dim=2), 1)
        idx2item = self.r_l_map
        lines = list()
        for predict, target, sentence in zip(batch_decoded, batch_targets, feature.sentences):
            gold = target % len(self.l_map)
            length = utils.find_length_from_labels(gold, self.l_map)
            predict = predict[:length].numpy()
            gold = gold[:length].numpy()
            sentence = sentence.tokens[:length]
            for i in range(length):
                # lines.append(f'{sentence[i]} '
                #              f'{idx2item[predict[i]]} '
                #              f'{idx2item[gold[i]]}\n')
                fout.write(f'{sentence[i]} '
                           f'{idx2item[predict[i]]} '
                           f'{idx2item[gold[i]]} '
                           f'\n')
            fout.write('\n')

    def call_conlleval(self, prefix):
        file_path = self.file_path / f'{prefix}.log'
        file_path_to = self.file_path / f'{prefix}.BIO'
        tagSchemeConvert = subprocess.check_output(f'python tools/convertResultTagScheme.py {file_path} {file_path_to}',
                                                   shell=True,
                                                   timeout=200)
        output = subprocess.check_output(f'perl tools/conlleval < {file_path_to}',
                                         shell=True,
                                         timeout=200).decode('utf-8')
        # if 'train' in prefix:
        if 'test' in prefix:
            pass
        else:
            delete = subprocess.check_output(f'rm -rf {file_path_to} {file_path}',
                                             shell=True,
                                             timeout=200).decode('utf-8')
        out = output.split('\n')[1]
        assert out.startswith('accuracy'), "Wrong lines"
        result = re.findall(r"\d+\.?\d*", out)
        return float(result[-1]), float(result[1]), float(result[2]), None

    def acc_score(self, *args):
        """
        calculate accuracy score based on statics
        """
        if 0 == self.total_labels:
            return 0.0
        accuracy = float(self.correct_labels) / self.total_labels
        return accuracy


class eval_w(eval_batch):
    """evaluation class for word level model (LSTM-CRF)

    args:
        packer: provide method to convert target into original space [TODO: need to improve]
        l_map: dictionary for labels
        score_type: use f1score with using 'f'

    """

    def __init__(self, l_map, score_type, file_path):
        eval_batch.__init__(self, l_map, file_path)

        self.decoder = CRFDecode_vb(len(l_map), l_map['<START>'], l_map['<PAD>'])

        self.eval_method = score_type
        if 'f' in score_type:
            # self.eval_b = self.calc_f1_batch
            # self.calc_s = self.f1_score
            self.eval_b = self.write_result
            self.calc_s = self.call_conlleval
        else:
            self.eval_b = self.calc_acc_batch
            self.calc_s = self.acc_score

    def calc_score(self, ner_model, dataset_loader, file_prefix=None):
        """
        calculate score for pre-selected metrics

        args:
            ner_model: LSTM-CRF model
            dataset_loader: loader class for test set
        """
        if 'f' in self.eval_method:
            fout = open(self.file_path / f'{file_prefix}.log', 'w')
        else:
            fout = None

        with torch.no_grad():

            ner_model.eval()
            self.reset()
            for batch in itertools.chain.from_iterable(dataset_loader):
                # mask = mask.transpose(0, 1).cuda()

                tg, mask = batch.tags, batch.masks
                # mask = mask.transpose(0, 1).cuda()
                score, _ = ner_model.forward(batch)
                decoded = self.decoder.decode(score.data, mask.data)
                self.eval_b(decoded, tg, batch, fout)

        if 'f' in self.eval_method:
            fout.close()
        return self.calc_s(file_prefix)


class eval_sec_crf(eval_batch):
    """evaluation class for word level model (LSTM-CRF)

    args:
        packer: provide method to convert target into original space [TODO: need to improve]
        l_map: dictionary for labels
        score_type: use f1score with using 'f'

    """

    def __init__(self, l_map, score_type, file_path):
        eval_batch.__init__(self, l_map, file_path)

        self.decoder = CRFSECDecode_vb(len(l_map), l_map['<START>'], l_map['<PAD>'])

        self.eval_method = score_type
        if 'f' in score_type:
            # self.eval_b = self.calc_f1_batch
            # self.calc_s = self.f1_score
            self.eval_b = self.write_result
            self.calc_s = self.call_conlleval
        else:
            self.eval_b = self.calc_acc_batch
            self.calc_s = self.acc_score

    def calc_score(self, ner_model, dataset_loader, file_prefix=None):
        """
        calculate score for pre-selected metrics

        args:
            ner_model: LSTM-CRF model
            dataset_loader: loader class for test set
        """
        if 'f' in self.eval_method:
            fout = open(self.file_path / f'{file_prefix}.log', 'w')
        else:
            fout = None

        with torch.no_grad():

            ner_model.eval()
            self.reset()
            for batch in itertools.chain.from_iterable(dataset_loader):
                # mask = mask.transpose(0, 1).cuda()

                tg, mask = batch.tags, batch.masks
                # mask = mask.transpose(0, 1).cuda()
                score, _ = ner_model.forward(batch)
                decoded = self.decoder.decode(score.data, mask.data)
                self.eval_b(decoded, tg, batch, fout)

        if 'f' in self.eval_method:
            fout.close()
        return self.calc_s(file_prefix)


# softmax
class eval_softmax(eval_batch):
    """evaluation class for word level model (LSTM-SOFTMAX)

    args:
        packer: provide method to convert target into original space [TODO: need to improve]
        l_map: dictionary for labels
        score_type: use f1score with using 'f'

    """

    def __init__(self, l_map, score_type, file_path=None):
        eval_batch.__init__(self, l_map, file_path)
        self.pad = l_map['<PAD>']
        self.eval_method = score_type
        if 'f' in score_type:
            # self.eval_b = self.calc_f1_batch
            # self.calc_s = self.f1_score
            self.eval_b = self.write_result
            self.calc_s = self.call_conlleval
        else:
            self.eval_b = self.calc_acc_batch
            self.calc_s = self.acc_score

    def decode(self, scores, masks, pad_tag):
        _, tags = torch.max(scores, 2)
        masks = ~masks
        tags.masked_fill_(masks, pad_tag)

        return tags.cpu()

    def calc_score(self, ner_model, dataset_loader, file_prefix=None):
        """
        calculate score for pre-selected metrics

        args:
            ner_model: LSTM-CRF model
            dataset_loader: loader class for test set
        """
        if 'f' in self.eval_method:
            fout = open(self.file_path / f'{file_prefix}.log', 'w')
        else:
            fout = None

        with torch.no_grad():
            ner_model.eval()
            self.reset()

            for batch in itertools.chain.from_iterable(dataset_loader):
                tg, mask = batch.tags, batch.masks
                # mask = mask.transpose(0, 1).cuda()
                score, _ = ner_model.forward(batch)
                decoded = self.decode(score.data, mask.cuda().data, self.pad)
                self.eval_b(decoded, tg, batch, fout)

        if 'f' in self.eval_method:
            fout.close()
        return self.calc_s(file_prefix)

    def calc_acc_batch(self, decoded_data, target_data, *args):
        """
        update statics for accuracy

        args:
            decoded_data (batch_size, seq_len): prediction sequence
            target_data (batch_size, seq_len): ground-truth
        """
        batch_decoded = torch.unbind(decoded_data, 1)
        batch_targets = torch.unbind(torch.squeeze(target_data.cpu()).transpose(0, 1), 0)

        for decoded, target in zip(batch_decoded, batch_targets):
            gold = target
            # remove padding
            length = utils.find_length_from_softmax_labels(gold, self.l_map)
            gold = gold[:length].numpy()
            best_path = decoded[:length].numpy()

            self.total_labels += length
            self.correct_labels += np.sum(np.equal(best_path, gold))

    def write_result(self, decoded_data, target_data, feature, fout):

        batch_decoded = torch.unbind(decoded_data, 1)
        batch_targets = torch.unbind(torch.squeeze(target_data.cpu(), dim=2), 0)
        idx2item = self.r_l_map
        lines = list()
        for predict, target, sentence in zip(batch_decoded, batch_targets, feature.sentences):
            tokens = sentence.tokens
            gold = target % len(self.l_map)
            length = utils.find_length_from_softmax_labels(gold, self.l_map)
            predict = predict[:length].numpy()
            gold = gold[:length].numpy()
            tokens = tokens[:length]
            for i in range(length):
                # lines.append(f'{sentence[i]} '
                #              f'{idx2item[predict[i]]} '
                #              f'{idx2item[gold[i]]}\n')
                fout.write(f'{tokens[i]} '
                           f'{idx2item[predict[i]]} '
                           f'{idx2item[gold[i]]} '
                           f'\n')
            fout.write('\n')


class Evaluator(object):
    def __init__(self, metric: str = None, path: str = None, itol=None, is_transfer=True, if_save=False):
        super(Evaluator, self).__init__()
        self.metric = metric
        self.path = path
        self.itol = itol
        self.is_transfer = is_transfer
        self.if_save = if_save
        if 'f1' in self.metric:
            self.cal_batch = self.cal_f1
        else:
            self.cal_batch = self.cal_acc
            self.correct_encoder = 0.0
            self.correct_q = 0.0
            self.count = 0.0

        self.QP_TT = 0.0
        self.QP_TF = 0.0
        self.QP_FT = 0.0
        self.QP_FF = 0.0
        self.total = 0.0

    def reset_path(self, path):
        self.path = path

    def reset_acc(self):
        if 'f1' not in self.metric:
            self.correct_encoder = 0.0
            self.correct_q = 0.0
            self.count = 0.0

    def reset_QP(self):
        self.QP_TT = 0.0
        self.QP_TF = 0.0
        self.QP_FT = 0.0
        self.QP_FF = 0.0
        self.total = 0.0

    def cal_confuse(self, QP, QG, length):
        # for i in range(length):
        self.total += length
        for qp, qg in zip(QP, QG):
            if qp:
                if qg: self.QP_TT += 1
                else: self.QP_FF += 1
            else:
                if qg: self.QP_TF += 1
                else: self.QP_FT += 1


    def cal_score(self, model, data_iter, prefix, device):
        model.eval()
        self.reset_acc()
        self.reset_QP()
        if 'f1' in self.metric:
            fout_q = open(self.path / f'{prefix}_q.txt', 'w')
            fout_e = open(self.path / f'{prefix}_e.txt', 'w')
            # fout = open(self.path / f'{prefix}.txt', 'w')
        elif self.if_save:
            fout_q = open(self.path / f'{prefix}_q.txt', 'w')
            fout_e = None
            # fout = None
        else:
            fout_q = None
            fout_e = None
            fout = None
        with torch.no_grad():
            for index, batch in enumerate(data_iter):
                # encoder_score, decoders_scores = model(batch.word[0],
                #                                        batch.word[1])
                encoder_score, decoders_scores = model(batch.bert[0],
                                                       batch.bert[1])

                if self.is_transfer:
                    decoders_labels = list()
                    for i in range(len(decoders_scores)):
                        exec(f'decoders_labels.append(batch.label{i + 1}.to(device))')
                    q_value = model.q_value(encoder_score, decoders_scores, decoders_labels)
                    # tags = model.decode(encoder_score, decoders_scores, batch.label.to(device))
                    tags = model.decode(q_value).cpu()
                    tags_encoder = model.decode(encoder_score).cpu()
                    self.cal_batch(batch.label, tags, tags_encoder, batch.bert[1], batch.bert[3], fout_q, fout_e)
                    # self.cal_batch(batch.label, tags, tags_encoder, batch.bert[1], batch.bert[3], fout_q, fout_e, decoders_labels, fout)
                else:
                    tags = model.decode(encoder_score).cpu()
                    tags_encoder = tags
                    self.cal_batch(batch.label, tags, tags_encoder, batch.bert[1], batch.bert[3], fout_q)
                # tags = model.decode(decoders_scores[0][:,:,:,0]).cpu()
                # if self.if_save:
                # else:
                # self.cal_batch(batch.label, tags, batch.word[1], batch.word[0], fout)
        #
        if 'f1' in self.metric:
            fout_q.close()
            fout_e.close()
            # fout.close()
        elif self.if_save:
            fout_q.close()
        score = self.final_score(prefix)

        return score

    def cal_acc(self, golds, predicts_q, predicts_e, lengths, sentences, fout, *args):
        golds = torch.unbind(golds, 0)
        predicts_q = torch.unbind(predicts_q, 0)
        predicts_e = torch.unbind(predicts_e, 0)
        lengths = torch.unbind(lengths.cpu(), 0)
        for gold, predict_q, predict_e, length, sentence in zip(golds, predicts_q, predicts_e, lengths, sentences):
            predict_q = predict_q[:length].numpy()
            predict_e = predict_e[:length].numpy()
            gold = gold[:length].numpy()
            tokens = sentence[1:length + 1]
            for i in range(length.numpy()):
                self.count += 1
                if predict_q[i] == gold[i]:
                    self.correct_q += 1
                if predict_e[i] == gold[i]:
                    self.correct_encoder += 1
                if self.if_save:
                    fout.write(f'{tokens[i]}\t'
                               f'{self.itol[predict_e[i]]}\t'
                               f'{self.itol[gold[i]]}\t'
                               f'\n')
            if self.if_save:
                fout.write('\n')
            # for p, g in zip(predict, gold):

    def cal_f1(self, golds, predicts, predicts_e, lengths, sentences, fout, fout_e=None):
    # def cal_f1(self, golds, predicts, predicts_e, lengths, sentences, fout, fout_e=None, labels=None, fout_=None):
    #     labels1 = torch.unbind(labels[0].cpu(), 1)
    #     labels2 = torch.unbind(labels[1].cpu(), 1)
    #     labels3 = torch.unbind(labels[2].cpu(), 1)
        golds = torch.unbind(golds, 0)
        predicts = torch.unbind(predicts, 0)
        predicts_e = torch.unbind(predicts_e, 0)
        lengths = torch.unbind(lengths.cpu(), 0)
        # sentences = torch.unbind(sentences.cpu(), 1)

        for gold, predict, predict_e, length, sentence in zip(golds, predicts, predicts_e, lengths, sentences):
        # for gold, predict, predict_e, length, sentence, label1, label2, label3 in zip(golds, predicts, predicts_e, lengths, sentences, labels1, labels2, labels3):
        #     label1 = label1[:length].numpy()
        #     label2 = label2[:length].numpy()
        #     label3 = label3[:length].numpy()

            predict = predict[:length].numpy()
            gold = gold[:length].numpy()
            tokens = sentence[1:length + 1]
            # tokens = sentence[:length]
            for i in range(length.numpy()):
                fout.write(f'{tokens[i]} '
                           f'{self.itol[predict[i]]} '
                           f'{self.itol[gold[i]]} '
                           f'\n')
            fout.write('\n')

            if fout_e is not None:
                predict_e = predict_e[:length].numpy()
                QP = predict == predict_e
                QG = predict == gold
                self.cal_confuse(QP, QG, length.numpy())
                for i in range(length.numpy()):
                    fout_e.write(f'{tokens[i]} '
                                 f'{self.itol[predict_e[i]]} '
                                 f'{self.itol[gold[i]]} '
                                 f'\n')
                fout_e.write('\n')

            # if fout_ is not None:
            #     # predict_e = predict_e[:length].numpy()
            #     for i in range(length.numpy()):
            #         fout_.write(f'{tokens[i]} '
            #                     f'{self.itol[label1[i]]} '
            #                     f'{self.itol[label2[i]]} '
            #                     f'{self.itol[label3[i]]} '
            #                     f'{self.itol[predict_e[i]]} '
            #                     f'\n')
            #     fout_.write('\n')

    def final_score(self, prefix):
        f1_e = None
        if 'f1' not in self.metric:
            return round(self.correct_q * 100 / self.count, 2), round(self.correct_encoder * 100 / self.count,
                                                                      2), None, None
        file_path = self.path / f'{prefix}_q.txt'
        file_path_to = file_path
        # file_path_to = self.path / f'{prefix}_q.BIO'
        # tagSchemeConvert = subprocess.check_output(f'python tools/convertResultTagScheme.py {file_path} {file_path_to}',
        #                                            shell=True,
        #                                            timeout=200)
        output = subprocess.check_output(f'perl tools/conlleval < {file_path_to}',
                                         shell=True,
                                         timeout=200).decode('utf-8')
        # if 'train' in prefix:
        # if 'test' in prefix:
        #     pass
        # else:
        #     delete = subprocess.check_output(f'rm -rf {file_path_to} {file_path}',
        #                                          shell=True,
        #                                          timeout=200).decode('utf-8')
        if not self.if_save:
            delete = subprocess.check_output(f'rm -rf {file_path_to} {file_path}',
                                             shell=True,
                                             timeout=200).decode('utf-8')
        # else:
        #     delete = subprocess.check_output(f'rm -rf {file_path_to}',
        #                                      shell=True,
        #                                      timeout=200).decode('utf-8')
        out = output.split('\n')[1]
        assert out.startswith('accuracy'), "Wrong lines"
        result = re.findall(r"\d+\.?\d*", out)

        if self.is_transfer:
            file_path_e = self.path / f'{prefix}_e.txt'
#             file_path_to_e = self.path / f'{prefix}_e.BIO'
#             tagSchemeConvert_e = subprocess.check_output(
#                 f'python tools/convertResultTagScheme.py {file_path_e} {file_path_to_e}',
#                 shell=True,
#                 timeout=200)
            output_e = subprocess.check_output(f'perl tools/conlleval < {file_path_e}',
                                               shell=True,
                                               timeout=200).decode('utf-8')
            # if 'train' in prefix:
            # if 'test' in prefix:
            #     pass
            # else:
            #     delete = subprocess.check_output(f'rm -rf {file_path_to} {file_path}',
            #                                          shell=True,
            #                                          timeout=200).decode('utf-8')
            if not self.if_save:
                delete = subprocess.check_output(f'rm -rf {file_path_e}',
                                                 shell=True,
                                                 timeout=200).decode('utf-8')
            else:
                delete = subprocess.check_output(f'rm -rf {file_path_e}',
                                                 shell=True,
                                                 timeout=200).decode('utf-8')
            out_e = output_e.split('\n')[1]
            assert out_e.startswith('accuracy'), "Wrong lines"
            result_e = re.findall(r"\d+\.?\d*", out_e)
            f1_e = float(result_e[-1])
        return float(result[-1]), f1_e, None, None


class Evaluator_finetune(Evaluator):
    def __init__(self, metric: str = None, path: str = None, itol=None, is_transfer=True, if_save=False):
        super(Evaluator_finetune, self).__init__(metric=metric, path=path, itol=itol, is_transfer=is_transfer, if_save=if_save)
        self.metric = metric
        self.path = path
        self.itol = itol
        self.is_transfer = is_transfer
        self.if_save = if_save
        self.scores = []
        if 'f1' in self.metric:
            self.cal_batch = self.cal_f1
        else:
            self.cal_batch = self.cal_acc
            self.correct_encoder = 0.0
            self.correct_q = 0.0
            self.count = 0.0

    def decode_sample(self):
        s = torch.nn.Softmax(dim=-1)
        prob = 0
        from itertools import product
        as1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        as2 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        as3 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        best = 0
        best_decode = None
        for a1, a2, a3 in product(as1, as2, as3):
            common = 0
        # for score in self.scores:
            prob = s(self.scores[0]) * a1 + s(self.scores[1]) * a2 + s(self.scores[2]) * a3
            tags = torch.argmax(prob, -1).cpu().numpy()
            pred = [self.itol[t] for t in tags]
            for p, g in zip(pred[:7], ['B-LOC', 'O', 'O', 'B-PER', 'O', 'O', 'O']):
                if p == g:
                    common += 1
            if best <= common:
                best = common
                best_decode = pred[:7]
        print(best_decode)

    def cal_score(self, model, data_iter, prefix, device):
        model.eval()
        self.reset_acc()
        if 'f1' in self.metric:
            fout_q = open(self.path / f'{prefix}_q.txt', 'w')
            fout_e = open(self.path / f'{prefix}_e.txt', 'w')
            # fout = open(self.path / f'{prefix}.txt', 'w')
        elif self.if_save:
            fout_q = open(self.path / f'{prefix}_q.txt', 'w')
            fout_e = None
            # fout = None
        else:
            fout_q = None
            fout_e = None
            fout = None
        with torch.no_grad():
            for index, batch in enumerate(data_iter):
                # encoder_score, decoders_scores = model(batch.word[0],
                #                                        batch.word[1])
                encoder_score, decoders_scores = model(batch.bert[0],
                                                       batch.bert[1])
                # for s in batch.bert[0]:
                #     if 'LOCKERBIE' in s and 'test_1071' in s:
                #         a = 3
                #         ind = batch.bert[0].index(s)
                #         print(encoder_score[ind])
                #         self.scores.append(encoder_score[ind][:19])

                if self.is_transfer:
                    decoders_labels = list()
                    for i in range(len(decoders_scores)):
                        exec(f'decoders_labels.append(batch.label{i + 1}.to(device))')
                    q_value = model.q_value(encoder_score, decoders_scores, decoders_labels)
                    # tags = model.decode(encoder_score, decoders_scores, batch.label.to(device))
                    tags = model.decode(q_value).cpu()
                    tags_encoder = model.decode(encoder_score).cpu()
                    self.cal_batch(batch.label, tags, tags_encoder, batch.bert[1], batch.bert[3], fout_q, fout_e)
                    # self.cal_batch(batch.label, tags, tags_encoder, batch.bert[1], batch.bert[3], fout_q, fout_e, decoders_labels, fout)
                else:
                    tags = model.decode(encoder_score).cpu()
                    tags_encoder = tags
                    self.cal_batch(batch.label, tags, tags_encoder, batch.bert[1], batch.bert[3], fout_q)
                # tags = model.decode(decoders_scores[0][:,:,:,0]).cpu()
                # if self.if_save:
                # else:
                # self.cal_batch(batch.label, tags, batch.word[1], batch.word[0], fout)
        #
        if 'f1' in self.metric:
            fout_q.close()
            fout_e.close()
            # fout.close()
        elif self.if_save:
            fout_q.close()
        score = self.final_score(prefix)

        return score

    def cal_acc(self, golds, predicts_q, predicts_e, lengths, sentences, fout, *args):
        golds = torch.unbind(golds, 0)
        predicts_q = torch.unbind(predicts_q, 0)
        predicts_e = torch.unbind(predicts_e, 0)
        lengths = torch.unbind(lengths.cpu(), 0)
        for gold, predict_q, predict_e, length, sentence in zip(golds, predicts_q, predicts_e, lengths, sentences):
            predict_q = predict_q[:length].numpy()
            predict_e = predict_e[:length].numpy()
            gold = gold[:length].numpy()
            tokens = sentence[1:length + 1]
            for i in range(length.numpy()):
                self.count += 1
                if predict_q[i] == gold[i]:
                    self.correct_q += 1
                if predict_e[i] == gold[i]:
                    self.correct_encoder += 1
                if self.if_save:
                    fout.write(f'{tokens[i]}\t'
                               f'{self.itol[predict_e[i]]}\t'
                               f'{self.itol[gold[i]]}\t'
                               f'\n')
            if self.if_save:
                fout.write('\n')
            # for p, g in zip(predict, gold):

    def cal_f1(self, golds, predicts, predicts_e, lengths, sentences, fout, fout_e=None):
    # def cal_f1(self, golds, predicts, predicts_e, lengths, sentences, fout, fout_e=None, labels=None, fout_=None):
    #     labels1 = torch.unbind(labels[0].cpu(), 1)
    #     labels2 = torch.unbind(labels[1].cpu(), 1)
    #     labels3 = torch.unbind(labels[2].cpu(), 1)
        golds = torch.unbind(golds, 0)
        predicts = torch.unbind(predicts, 0)
        predicts_e = torch.unbind(predicts_e, 0)
        lengths = torch.unbind(lengths.cpu(), 0)
        # sentences = torch.unbind(sentences.cpu(), 1)

        for gold, predict, predict_e, length, sentence in zip(golds, predicts, predicts_e, lengths, sentences):
        # for gold, predict, predict_e, length, sentence, label1, label2, label3 in zip(golds, predicts, predicts_e, lengths, sentences, labels1, labels2, labels3):
        #     label1 = label1[:length].numpy()
        #     label2 = label2[:length].numpy()
        #     label3 = label3[:length].numpy()

            predict = predict[:length].numpy()
            gold = gold[:length].numpy()
            tokens = sentence[1:length + 1]
            # tokens = sentence[:length]
            for i in range(length.numpy()):
                fout.write(f'{tokens[i]} '
                           f'{self.itol[predict[i]]} '
                           f'{self.itol[gold[i]]} '
                           f'\n')
            fout.write('\n')

            if fout_e is not None:
                predict_e = predict_e[:length].numpy()
                for i in range(length.numpy()):
                    fout_e.write(f'{tokens[i]} '
                                 f'{self.itol[predict_e[i]]} '
                                 f'{self.itol[gold[i]]} '
                                 f'\n')
                fout_e.write('\n')

            # if fout_ is not None:
            #     # predict_e = predict_e[:length].numpy()
            #     for i in range(length.numpy()):
            #         fout_.write(f'{tokens[i]} '
            #                     f'{self.itol[label1[i]]} '
            #                     f'{self.itol[label2[i]]} '
            #                     f'{self.itol[label3[i]]} '
            #                     f'{self.itol[predict_e[i]]} '
            #                     f'\n')
            #     fout_.write('\n')

