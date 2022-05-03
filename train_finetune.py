import argparse
from configparser import ConfigParser
from pathlib import  Path
from typing import List, Tuple, Dict, Set
import shutil
import sys
import time
from models.utils import (
    configparser,
    log_info,
)
from models.params import Params
from models.crf import *
from models.lstm_model import *
from models.evaluator import *
from models.transfer_model import *
from models.utils import (
    bert_pad
)
import torchtext
from torchtext.vocab import (
    FastText, GloVe
)
from torchtext.data import (
    Field, NestedField, BucketIterator, Iterator
)
from processing.datasets import (
    SequenceData,
    BertField,
    MyEmbeddings,
    QValueField
)
from processing.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharacterEmbeddings, BertEmbeddings

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import time


def check(words, bert_lengths, q_lengths, keys):
    for word, b_l, q_l, key in zip(words, bert_lengths, q_lengths, keys):
        assert word[0] == key, f'Wrong key, bert {word[0]} q {key}'
        assert b_l == q_l, f'Wrong length, bert {b_l}, q {q_l}'

def train(train_iter: Iterator,
          dev_iter: Iterator,
          test_iter: Iterator,
          network: nn.Module,
          optimizer: optim,
          device,
          Epoch: int,
          max_m_epoch: int=1,
          prev_epoch=0,
          pre_e_loss=10000,
          e_patience=0,
          evaluator=None,
          scheduler=None,
          decay_p=None,
          fout=None):
    # Calculate and reset QValue
    network.eval()
    log_info('-' * 100)
    log_info(f'Epoch {Epoch} begin ...')
    start_time = time.time()
    with torch.no_grad():
        qValueField = train_iter.dataset.fields['qValue']
        e_loss = 0
        for index, batch in enumerate(train_iter):
            check(batch.bert[-1], batch.bert[1], batch.qValue[2], batch.qValue[1])
            decoders_labels = list()
            for i in range(params.num_decoders):
                exec(f'decoders_labels.append(batch.label{i+1}.to(device))')
            encoder_score, decoders_scores = network(batch.bert[0],
                                                     batch.bert[1], p=decay_p.p())

            q_value = network.q_value(encoder_score, decoders_scores, decoders_labels)
            loss = network.loss(q_value, encoder_score, decoders_scores, decoders_labels, batch.bert[2])
            e_loss += loss
            qValueField.reset_vocab(q_value, batch.qValue[1], batch.qValue[2])
        e_loss = e_loss / (index + 1)
        if e_loss >= pre_e_loss * 0.999:
            e_patience += 1
        else:
            pre_e_loss = e_loss
            e_patience = 0
    e_step_end = time.time()
    log_info(f'E Step loss {e_loss}, patience {e_patience} spend {e_step_end - start_time} sec.')

    # val_score, val_score_e, _, _ = evaluator.cal_score(network, dev_iter, 'val', device)
    # train_score, train_score_e, _, _ = evaluator.cal_score(network, train_iter, 'train', device)
    test_score, test_score_e, _, _ = evaluator.cal_score(network, test_iter, 'test', device)
    val_score, val_score_e = 0, 0
    log_info(f'val_score {val_score} test_score {test_score}')
    if params.use_transfer:
        log_info(
            f'val_score_encoder {val_score_e} test_score_encoder {test_score_e}')

        # log_info(f'{evaluator.QP_TT / evaluator.total}, {evaluator.QP_FT / evaluator.total},'
        #          f' {evaluator.QP_TF / evaluator.total}, {evaluator.QP_FF / evaluator.total}.')

    fout.writelines(f'Epoch {Epoch} E Step loss {e_loss:.4f} - val_score {val_score} test_score {test_score} - val_score_encoder {val_score_e} test_score_encoder {test_score_e}\n')
    if params.max_epochs <= Epoch:
        return prev_epoch + 1 + 0, pre_e_loss, e_patience, test_score
    network.train()
    epoch_loss = 0
    prev_loss = 10000
    patience = 0
    total_number_of_batches = train_iter.iterations
    modulo = max(1, int(total_number_of_batches / 10))
    batch_size = train_iter.batch_size
    for epoch in range(max_m_epoch):
        seen_batches = 0
        epoch_loss = 0
        # log_info('-' * 100)
        batch_time = 0
        for index, batch in enumerate(train_iter):
            check(batch.bert[-1], batch.bert[1], batch.qValue[2], batch.qValue[1])
            start_time = time.time()
            network.zero_grad()
            curr_q = batch.qValue[0]
            keys = batch.qValue[1]
            leng = batch.qValue[2]
            decoders_labels = list()
            for i in range(params.num_decoders):
                exec(f'decoders_labels.append(batch.label{i + 1}.to(device))')
            encoder_score, decoders_scores = network(batch.bert[0],
                                                     batch.bert[1], p=decay_p.p())

            loss = network.loss(curr_q, encoder_score, decoders_scores, decoders_labels, batch.bert[2])
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()
            # network.reset_vectors()
            epoch_loss += loss
            seen_batches += 1

            batch_time += time.time() - start_time
            if seen_batches % modulo == 0:
                p = decay_p.p()
                log_info(
                    f"M step epoch {epoch + 1} - iter {seen_batches}/{total_number_of_batches} - loss "
                    f"{epoch_loss / seen_batches:.8f} - {p:.8f} - "
                    f"samples/sec: {batch_size * modulo / batch_time:.2f}",
                    dynamic=True
                )
                batch_time = 0
            scheduler.step()
            decay_p.step()
        epoch_loss = epoch_loss / (index + 1)
        # log_info(f'loss: {epoch_loss}.', dynamic=False)

    log_info(f'M Step spend {time.time() - e_step_end} sec.')

    return prev_epoch + 1 + epoch, pre_e_loss, e_patience, test_score


def evaluate():

    return None


def construct_data(params: Params, device):
    # word: Field = Field(eos_token='<eos>', pad_token='<pad>', include_lengths=True)
    bert: Field = BertField(device=device, use_vocab=False, pad_token='<pad>', include_lengths=True, batch_first=True)
    label: Field = Field(pad_token='<pad>', batch_first=True)
    qValue: Field = QValueField(device=device, use_vocab=True,
                                postprocessing=bert_pad, include_lengths=True)

    # fields: List = [(('word', 'char'), (word, char)), ('label', label)]
    fields: List = [(('bert', 'qValue'), (bert, qValue))]
    for i in range(params.num_decoders):
        fields.append((f'label{i+1}', label))
    fields.append(('label', label))
    separator = '\t' if params.task in ['pos', 'POS'] else ' '
    train_set, val_set, test_set = SequenceData.splits(fields=fields, root=params.root, task=params.task,
                                                       corpora=params.corpus, separator=separator)

    data_config = torch.load(Path(params.root) / f'config_{params.task}.pt')
    # if params.task == 'TechNews':
    #     label.vocab = data_config[f'CoNLL']['label_dic']
    #     # data_config[f'{params.domain}']['label_dic'] = label.vocab
    #     torch.save({
    #         f'{params.domain}':
    #             {'label_dic': label.vocab}
    #     }, Path(params.root) / f'config_{params.task}.pt')
    #     print(label.vocab)
    #     exit(0)
    label.vocab = data_config[f'{params.domain}']['label_dic']

    # label.build_vocab(train_set.label)
    label_dict = label.vocab
    print(label_dict.stoi)
    qValue.build_vocab(len(label_dict), train_set.qValue, val_set.qValue, test_set.qValue)

    assert (params.bert != 'None') and (params.bert != ' ') and (params.bert != None), \
        f"No such BERT: {params.bert}"
    sys.stdout.flush()

    # train_iter = BucketIterator.splits(
    #     (train_set), batch_size=(params.batch), sort_key=lambda x: len(x.text), shuffle=True)
    # train_set.examples = train_set.examples[:1000]
    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train_set, val_set, test_set), batch_sizes=(params.batch, 50, 50), sort_key=lambda x: len(x.bert))

    log_info('-' * 100)
    log_info('-' * 100)
    log_info(params.corpus)
    log_info(f'Corpus: "train {len(train_set)} val {len(val_set)} test {len(test_set)}"')
    log_info('-' * 100)
    log_info(f' - mini_batch_size: "{params.batch}"')

    return (label_dict, train_iter, val_iter, test_iter)


def run(params: Params, data: Tuple, device: str, ro=None):
    (label_vocab, train_iter, dev_iter, test_iter) = data

    label_dict = label_vocab.stoi
    eval_dict = label_vocab.itos
    transfer_model = Transfer_Model(label_dict, params.bert,
                                    params.dropout, params.model_type, params.num_decoders,
                                    params.corpus, params.softem, params.mu, device=device, method=params.method,
                                    init_mu=params.init_mu)
    transfer_model.rand_init()
    if params.load_model:
        transfer_model.load_source_model(params.load_model_path)

    diff_lr = params.method == 'lvm'
    if diff_lr:
        model_params = []
        for k, v in dict(transfer_model.named_parameters()).items():
            if 'label_embeddings' in k:
                model_params.append({'params': [v], 'lr': params.HP_top_lr})
                log_info(f'set {params.HP_top_lr} lr  to {k}.')
            else:
                model_params.append({'params': [v], 'lr': params.HP_lr})
        optimizer = AdamW(model_params, lr=params.HP_lr, betas=(0.9, 0.999), weight_decay=params.L2)
        # optimizer = optim.SGD(model_params, weight_decay=params.L2)
        log_info(f'Diff learning rate: lr: {params.HP_lr} - top_lr: {params.HP_top_lr}')
    else:
        optimizer = AdamW(transfer_model.parameters(), lr=params.HP_lr, betas=(0.9, 0.999), weight_decay=params.L2)

    transfer_model.to(device)

    # TODO: change results dirs
    model_dir = Path(params.model_path) / params.table
    # log_info(f'Model: "{transfer_model}"')
    log_info('-' * 100)
    log_info("Parameters:")
    log_info(f' - learning_rate: "{params.HP_lr}"')
    log_info(f' - patience: "{params.patience}"')
    log_info(f' - max_epochs: "{params.max_epochs}"')
    log_info(f' - mu: {params.mu}')
    log_info('-' * 100)
    log_info(f'Model training base path: "{model_dir}"')
    log_info('-' * 100)
    # log.info(f"Device: {device}")
    log_info('-' * 100)

    best_score = float('-inf')
    test_score = float('inf')
    track_list = list()
    epoch_list = range(0, params.max_epochs + 1)
    patience_count = 0

    if params.mode in 'tune':
        model_path = model_dir / f'{params.HP_lr}_{params.hidden_size}_{params.HP_tag_dim}_{params.HP_rank}'
    else: model_path = model_dir / params.method
    if not (os.path.exists(model_path) and os.path.isdir(model_path)):
        os.makedirs(model_path, exist_ok=True)
    # if params.use_crf:
    #     evaluator = eval_w(label_dict, params.metric, model_path)
    # else:
    #     evaluator = eval_softmax(label_dict, params.metric, model_path)

    log_info(str(model_path))
    log_info(f'round={ro} method={params.method} lr={params.HP_lr} top_lr={params.HP_top_lr} mu={params.mu} init_mu={params.init_mu} max-epoch={params.max_epochs}\n')
    write_log = model_path / 'result.log'
    fout = open(write_log, 'a')
    fout.writelines(f'round={ro} method={params.method} lr={params.HP_lr} top_lr={params.HP_top_lr} mu={params.mu} init_mu={params.init_mu} max-epoch={params.max_epochs}\n')
    evaluator = Evaluator(params.metric, model_path, eval_dict, is_transfer=True, if_save=False)

    evaluator_source = Evaluator_finetune(params.metric, model_path, eval_dict, is_transfer=False, if_save=False)
    for source in transfer_model.decoder_nets:
        m = source.model
        m.eval()
        test_score, test_score_e, _, _ = evaluator_source.cal_score(m, test_iter, 'test', device)
        log_info(f'test_score {test_score}')
    # exit(0)
    previous_learning_rate = params.HP_lr

    # scheduler = StepLR(optimizer, step_size=1, gamma=0.8)

    total_number_of_batches = len(train_iter)
    modulo = max(1, int(total_number_of_batches / 10))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_number_of_batches * 0.1,
                                                num_training_steps=total_number_of_batches * params.max_epochs)
    decay_p = utils.decay_interpolation(total_number_of_batches * params.max_epochs)
    e = 0
    e_loss = 10000
    e_patience = 0
    for epoch_idx, start in enumerate(epoch_list):
        # get new learning rate
        for group in optimizer.param_groups:
            learning_rate = group["lr"]

        log_info(f'learning_rate: {learning_rate:8f}')

        e, e_loss, e_patience, test_score = train(train_iter, dev_iter, test_iter, transfer_model, optimizer, device, epoch_idx,
                                                  prev_epoch=e, pre_e_loss=e_loss, e_patience=e_patience, evaluator=evaluator,
                                                  scheduler=scheduler, decay_p=decay_p, fout=fout)

        # if e_patience >= 5: break
    fout.close()
    fig_path = params.task
    if 'fix' in params.table: return None
    s = torch.nn.Softmax(dim=0)

    try:
        matrixs = []
        o_matrixs = []
        M = []
        for d in transfer_model.decoder_nets:
            # print(d.label_embeddings)
            print(s(d.label_embeddings).cpu())
            m = d.label_embeddings.data.cpu().detach()
            # M.append(m.numpy())
            matrixs.append(s(m).numpy())
            o_matrixs.append(m.numpy())
        # plot_hot_fig(f'{params.table}_prob', eval_dict, matrixs, fig_path)
        # plot_hot_fig(f'{params.table}_table', eval_dict, o_matrixs, fig_path)
    except Exception as f:
        log_info(f)
        print(transfer_model.label_embeddings)
        print(s(transfer_model.label_embeddings).cpu())
        # plot_hot_fig(f'{params.table}_prob', eval_dict, [s(transfer_model.label_embeddings.cpu()).numpy()], fig_path)
        # plot_hot_fig(f'{params.table}_table', eval_dict, [transfer_model.label_embeddings.cpu().numpy()], fig_path)

    log_info(f'saving model... {str(model_path / "best_model.pt")}')
    # if params.save_model:
    if True:
        torch.save({
            'label': label_vocab,
            'embeddings': params.bert,
            'task': params.task,
            # 'source_corpus': params.source_corpus,
            'metric': params.metric,
            'model_state_dict': transfer_model.state_dict()
        }, model_path / 'best_model.pt')

    # run1(params, data, device, M)
    return test_score

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Learning with ncrf or triLinear-ncrf')
    parser.add_argument('--config', default='ner_3_de.config', help='Path of .config file.')
    parser.add_argument('--method', default='lvm', help='Use which approach?')
    parser.add_argument('--table', default='multi', help='Use which approach?')
    parser.add_argument('--cluster', default='TitanV', help='Running on which cluster, et., AI, TitanV, and P40.')
    parser.add_argument('--free', type=int, default=0, help='GPU device')
    # a = os.listdir(Path('../.cache'))
    args = parser.parse_args()
    config_name = args.config
    cluster = args.cluster
    # torch.cuda.set_device(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 0. Optional: occupy memory of GPU before running tasks.
    if device.type not in 'cpu':
        free = args.free
        x = torch.IntTensor(256, 1024, free).to(device)
        print(f'Occupy {int(free)}, {x.size(-1)}')
        del x

    config_file = Path('./config/train') / Path(config_name)
    log_info(config_file)
    config = configparser()
    config.read(config_file, encoding='utf-8')
    # 1. Read experiments' settings.
    params = Params(config)
    setattr(params, 'method', args.method)
    if args.table != 'multi':
        params.table = args.table
    log_info(params.table)

    from itertools import product
    # 2. Construct data and embeddings.
    data = construct_data(params, device)
    if params.mode in 'train':
        # Each settings should run n rounds to average performance.
        from itertools import product
        mus = [2] #
        setattr(params, 'init_mu', None)
        init_mus = [10] # [10, 4, 3, 2, 1] for onto and ner and pos
        # init_mus = [10, 3, 2, 1] # for sem
        lrs = [2e-5]# [2e-5, 3e-5, 5e-5]
        max_epochs = [5] #[3, 5, 7]
        if params.method == 'mrt':
            top_lrs = [0.]
            mus = [init_mus[0]]
        else:
            top_lrs = [2e-4] #[2e-3, 2e-4]#
        for lr, max_epoch, top_lr, mu, init_mu in product(lrs, max_epochs, top_lrs, mus, init_mus):
            if params.method == 'mrt':
                mu = init_mu
            params.reset(mu, top_lr, lr, max_epoch, init_mu)
            for round_i in range(5):
                res = run(params, data, device, ro=round_i)

