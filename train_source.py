import argparse
from configparser import ConfigParser
from pathlib import Path
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
from processing.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharacterEmbeddings, \
    BertEmbeddings

import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import time


def check(words, bert_lengths, q_lengths, keys):
    for word, b_l, q_l, key in zip(words, bert_lengths, q_lengths, keys):
        assert word[0] == key, f'Wrong key, bert {word[0]} q {key}'
        assert b_l == q_l, f'Wrong length, bert {b_l}, q {q_l}'


def train(train_iter: Iterator,
          network: nn.Module,
          optimizer: optim,
          device,
          Epoch: int,
          max_m_epoch: int = 1,
          scheduler: get_linear_schedule_with_warmup = None):
    # Calculate and reset QValue
    network.train()
    log_info('-' * 100)
    start_time = time.time()

    total_number_of_batches = len(train_iter)#.iterations
    modulo = max(1, int(total_number_of_batches / 10))
    batch_size = train_iter.batch_size
    seen_batches = 0
    epoch_loss = 0
    # log_info('-' * 100)
    batch_time = 0
    for index, batch in enumerate(train_iter):
        start_time = time.time()
        network.zero_grad()

        scores, _ = network(batch.bert[0], batch.bert[1])

        loss = network.crit(scores, batch.label.to(device), batch.bert[2])
        loss.backward()
            # loss = network.crit(decoders_scores[0], batch.label.to(device), batch.bert[2])
            # loss = network.crit(encoder_score, decoders_scores, batch.label.to(device), batch.bert[2])
            # loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), 5.0)
        optimizer.step()
        epoch_loss += loss
        seen_batches += 1

        batch_time += time.time() - start_time
        if seen_batches % modulo == 0:
            log_info(
                f"epoch {Epoch + 1} - iter {seen_batches}/{total_number_of_batches} - loss "
                f"{epoch_loss / seen_batches:.8f} - samples/sec: {batch_size * modulo / batch_time:.2f}",
                dynamic=True
            )
            batch_time = 0
        scheduler.step()


    epoch_loss = epoch_loss / (index + 1)
    print(f'')
    log_info(f'loss: {epoch_loss}.', dynamic=False)

    return None


def evaluate_es(model_path, params):
    bert: Field = BertField(device=device, use_vocab=False, pad_token='<pad>', include_lengths=True, batch_first=True)
    label: Field = Field(pad_token='<pad>', batch_first=True)

    fields: List = [('bert', bert), ('label', label)]
    data_config = torch.load(Path(params.root) / f'config_{params.task}.pt')
    label.vocab = data_config[f'{params.corpus}']['label_dic']

    if_save = True# if params.mode != 'eval' else False
    evaluator = Evaluator_finetune(params.metric, model_path, label.vocab.itos, is_transfer=False, if_save=if_save)
    for source_name, source in zip(['en', 'de', 'nl'], ['conll_03_english', 'conll_03_german', 'conll_03_dutch']):
        model_path = Path(f'.transfer_data/ner/{source}/softmax_mbert/')
        eval_net = Softmax(model=params.bert, num_labels=len(label.vocab), dropout=0.1, device=device)
        model_data = torch.load(model_path / source_name / 'best_model.pt')
        eval_net.load_state_dict(model_data['model_state_dict'])

        eval_net.to(device)
        separator = '\t' if params.task in ['pos'] else ' '

        for target_corpus in params.target_corpus:
            if target_corpus != 'conll_03_spanish': continue
            log_info('-' * 100)
            log_info(f'target {target_corpus}')

            train_set, val_set, test_set = SequenceData.splits(fields=fields, root=params.root, task='',
                                                               corpora=target_corpus, separator=separator)
            log_info(f'Corpus: "train {len(train_set)} val {len(val_set)} test {len(test_set)}"')
            train_iter_t, val_iter_t, test_iter_t = Iterator.splits((train_set, val_set, test_set),
                                                                    batch_sizes=(50, 50, 50), shuffle=False)
            # bert.reset_vocab()
            # bert.build_vocab(Bert_model, train_set.bert, val_set.bert, test_set.bert)

            target_path = model_path / (params.source_name + '_' + target_corpus)
            if not (os.path.exists(target_path) and os.path.isdir(target_path)):
                os.makedirs(target_path, exist_ok=True)
            evaluator.reset_path(target_path)
            # train_score = evaluator.cal_score(eval_net, train_iter_t, 'train', device)
            # val_score = evaluator.cal_score(eval_net, val_iter_t, 'val', device)
            test_score = evaluator.cal_score(eval_net, test_iter_t, 'test', device)
            # log_info(f'train_score {train_score} val_score {val_score} test_score {test_score}')
            log_info(f'test_score {test_score}')
    evaluator.decode_sample()
    return None


def evaluate(model_path, params):
    model_data = torch.load(model_path / params.source_name / 'best_model.pt')
    bert: Field = BertField(device=device, use_vocab=False, pad_token='<pad>', include_lengths=True, batch_first=True)
    label: Field = Field(pad_token='<pad>', batch_first=True)

    fields: List = [('bert', bert), ('label', label)]
    data_config = torch.load(Path(params.root) / f'config_{params.task}.pt')
    label.vocab = data_config[f'{params.corpus}']['label_dic']

    eval_net = Softmax(model=params.bert, num_labels=len(label.vocab), dropout=0.1, device=device)

    eval_net.load_state_dict(model_data['model_state_dict'])
    if_save = True# if params.mode != 'eval' else False
    evaluator = Evaluator_finetune(params.metric, model_path, label.vocab.itos, is_transfer=False, if_save=if_save)

    eval_net.to(device)
    separator = '\t' if params.task in ['pos'] else ' '

    for target_corpus in params.target_corpus:
        log_info('-' * 100)
        log_info(f'target {target_corpus}')

        train_set, val_set, test_set = SequenceData.splits(fields=fields, root=params.root, task='',
                                                           corpora=target_corpus, separator=separator)
        log_info(f'Corpus: "train {len(train_set)} val {len(val_set)} test {len(test_set)}"')
        train_iter_t, val_iter_t, test_iter_t = Iterator.splits((train_set, val_set, test_set),
                                                                batch_sizes=(50, 50, 50), shuffle=False)
        # bert.reset_vocab()
        # bert.build_vocab(Bert_model, train_set.bert, val_set.bert, test_set.bert)

        target_path = model_path / (params.source_name + '_' + target_corpus)
        if not (os.path.exists(target_path) and os.path.isdir(target_path)):
            os.makedirs(target_path, exist_ok=True)
        evaluator.reset_path(target_path)
        train_score = evaluator.cal_score(eval_net, train_iter_t, 'train', device)
        val_score = evaluator.cal_score(eval_net, val_iter_t, 'val', device)
        test_score = evaluator.cal_score(eval_net, test_iter_t, 'test', device)
        log_info(f'train_score {train_score} val_score {val_score} test_score {test_score}')
        # log_info(f'test_score {test_score}')
    # evaluator.decode_sample()
    return None


def construct_data(params: Params, device):
    # word: Field = Field(eos_token='<eos>', pad_token='<pad>', include_lengths=True)
    bert: Field = BertField(device=device, use_vocab=False, pad_token='<pad>', include_lengths=True, batch_first=True)
    label: Field = Field(pad_token='<pad>', batch_first=True)

    # fields: List = [(('word', 'char'), (word, char)), ('label', label)]
    fields: List = [('bert', bert), ('label', label)]
    print(params.task, params.source_corpus, params.target_corpus)
    separator = '\t' if params.task in ['pos'] else ' '
    train_set, val_set, test_set = SequenceData.splits(fields=fields, root=params.root, task='',
                                                       corpora=params.source_corpus, separator=separator)
    
    # label.build_vocab(train_set.label)
    # torch.save({
    #     f'{params.corpus}':
    #         {'label_dic': label.vocab}
    # }, Path(params.root) / f'config_{params.task}.pt')
    # data_config = torch.load(Path(params.root) / f'config_{params.task}.pt')
    # tmp = data_config[f'{params.corpus}']['label_dic']
    # label.build_vocab(train_set.label)
    # log_info(label.vocab.stoi)
    # tmp.extend(label.vocab)
    # for target_corpus in params.target_corpus:
    #     train_set, val_set, test_set = SequenceData.splits(fields=fields, root=params.root, task='',
    #                                                        corpora=target_corpus, separator=separator)
    #     label.build_vocab(train_set.label)
    #     log_info(f'{target_corpus} {label.vocab.itos}')
    #     tmp.extend(label.vocab)
    #
    # torch.save({
    #     f'{params.corpus}':
    #         {'label_dic': tmp}
    # }, Path(params.root) / f'config_{params.task}.pt')
    # print(tmp.stoi)
    # exit(0)
    data_config = torch.load(Path(params.root) / f'config_{params.task}.pt')
    label.vocab = data_config[f'{params.corpus}']['label_dic']
    label_dict = label.vocab
    print(label_dict.stoi)

    assert (params.bert != 'None') and (params.bert != ' ') and (params.bert != None), \
        f"No such BERT: {params.bert}"
    sys.stdout.flush()

    # train_iter = BucketIterator.splits(
    #     (train_set), batch_size=(params.batch), sort_key=lambda x: len(x.text), shuffle=True)
    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train_set, val_set, test_set), batch_sizes=(params.batch, 16, 16), sort_key=lambda x: len(x.bert))

    log_info('-' * 100)
    log_info('-' * 100)
    log_info(f'Corpus: "train {len(train_set)} val {len(val_set)} test {len(test_set)}"')
    log_info('-' * 100)
    log_info(f' - mini_batch_size: "{params.batch}"')

    return (label_dict, train_iter, val_iter, test_iter)


def run(params: Params, data: Tuple, device: str):
    (label_vocab, train_iter, dev_iter, test_iter) = data

    label_dict = label_vocab.stoi
    eval_dict = label_vocab.itos
    net = Softmax(model=params.bert, num_labels=len(label_vocab), dropout=0.1, device=device)

    net_params = []
    if params.freeze:
        log_info(f'Freezing last three layers!')
        for name, value in dict(net.named_parameters()).items():
            if name.startswith('model.model.encoder.layer.0.') or name.startswith(
                    'model.model.encoder.layer.1.') or name.startswith('model.model.encoder.layer.2.'):
                net_params.append({'params': [value], 'lr': 0.0})
                log_info(f'\t{name}')
            elif 'model.model.embeddings' in name:
                net_params.append({'params': [value], 'lr': 0.0})
                log_info(f'\t{name}')
            else:
                net_params.append({'params': [value], 'lr': params.HP_lr})
    else:
        for name, value in dict(net.named_parameters()).items():
            if 'mapping_mat' in name or 'source_models.weight_vector' in name or 'combine_layer.bilinears' in name:
                net_params.append({'params': [value], 'lr': params.HP_top_lr})
                log_info(f'set {name} lr to {params.HP_top_lr}')
            else:
                net_params.append({'params': [value], 'lr': params.HP_lr})

    optimizer = AdamW(net_params, lr=params.HP_lr, betas=(0.9, 0.999))

    net.to(device)

    model_dir = Path(params.model_path)
    # log_info(f'Model: "{net}"')
    log_info('-' * 100)
    log_info("Parameters:")
    log_info(f' - learning_rate: "{params.HP_lr}"')
    log_info(f' - patience: "{params.patience}"')
    log_info(f' - max_epochs: "{params.max_epochs}"')
    log_info('-' * 100)
    log_info(f'Model training base path: "{model_dir}"')
    log_info('-' * 100)
    # log.info(f"Device: {device}")
    log_info('-' * 100)

    best_score = float('-inf')
    test_score = float('inf')
    epoch_list = range(0, params.max_epochs)

    if params.mode in 'tune':
        model_path = model_dir / f'{params.HP_lr}_{params.hidden_size}_{params.HP_tag_dim}_{params.HP_rank}'
    else:
        path = ''
        if params.use_crf:
            path += 'crf_'
        else:
            path += 'softmax_'
        if params.use_bert == 1:
            path += 'mbert'
        elif params.use_bert == 0:
            path += 'xlmr_large'
        elif params.use_bert == 2:
            path += 'xlmr'
        elif params.use_bert == 4:
            path += 'bert_large'
        elif params.use_bert == 5:
            path += 'bert_six'
        model_path = model_dir / path
    if not (os.path.exists(model_path) and os.path.isdir(model_path)):
        os.makedirs(model_path, exist_ok=True)
    source_path = model_path / params.source_name
    if not (os.path.exists(source_path) and os.path.isdir(source_path)):
        os.makedirs(source_path, exist_ok=True)
    # if params.use_crf:
    #     evaluator = eval_w(label_dict, params.metric, model_path)
    # else:
    #     evaluator = eval_softmax(label_dict, params.metric, model_path)
    log_info(params.metric)
    log_info(source_path)
    evaluator = Evaluator_finetune(params.metric, source_path, eval_dict, is_transfer=False, if_save=True)
    previous_learning_rate = params.HP_lr

    # max_epoch = 3
    total_number_of_batches = len(train_iter)
    modulo = max(1, int(total_number_of_batches / 10))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_number_of_batches * 0.1,
                                                num_training_steps=total_number_of_batches * params.max_epochs)


    for epoch_idx, start in enumerate(epoch_list):
        # get new learning rate
        for group in optimizer.param_groups:
            learning_rate = group["lr"]

        # stop training if learning rate becomes too small
        batch_time = 0
        log_info(f'learning_rate: {learning_rate:.8f}')

        train(train_iter, net, optimizer, device, epoch_idx, scheduler=scheduler)

        val_score, _, _, _ = evaluator.cal_score(net, dev_iter, 'val', device)
        if best_score < val_score:
            log_info(f'saving model... {str(source_path / "best_model.pt")}')
            if params.save_model:
                torch.save({
                    'label': label_vocab,
                    'embeddings': params.bert,
                    'task': params.task,
                    'source_corpus': params.source_corpus,
                    'metric': params.metric,
                    'model_state_dict': net.state_dict()
                }, source_path / 'best_model.pt')
            best_score = val_score
            test_score, _, _, _ = evaluator.cal_score(net, test_iter, 'test', device)

        log_info(f'val_score {val_score} test_score {test_score}')

    train_score, _, _, _ = evaluator.cal_score(net, train_iter, 'train', device)
    log_info(f'train_score {train_score} val_score {best_score} test_score {test_score}')

    # if params.mode == 'online':
    #     evaluate(model_path, params)
    # print(transfer_model.label_embeddings)

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning with ncrf or triLinear-ncrf')
    parser.add_argument('--config', default='ner_en.config', help='Path of .config file.')
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

    config_file = Path('./config/raw_train') / config_name
    config = configparser()
    config.read(config_file, encoding='utf-8')
    # 1. Read experiments' settings.
    params = Params(config)

    # 2. Construct data and embeddings.
    data = construct_data(params, device)
    if params.mode in ['train', 'online']:
        # Each settings should run n rounds to average performance.
        for round_i in range(params.rounds):
            # pass
            run(params, data, device)
        if params.mode == 'online':
            model_dir = Path(params.model_path)
            path = ''
            if params.use_crf:
                path += 'crf_'
            else:
                path += 'softmax_'
            if params.use_bert == 1:
                path += 'mbert'
            elif params.use_bert == 0:
                path += 'xlmr_large'
            elif params.use_bert == 2:
                path += 'xlmr'
            elif params.use_bert == 4:
                path += 'bert_large'
            elif params.use_bert == 5:
                path += 'bert_six'
            model_path = model_dir / path
            evaluate(model_path, params)
            # evaluate_es(model_path, params)
    elif params.mode in ['eval']:
        model_dir = Path(params.model_path)
        path = ''
        if params.use_crf:
            path += 'crf_'
        else:
            path += 'softmax_'
        if params.use_bert == 1:
            path += 'mbert'
        elif params.use_bert == 0:
            path += 'xlmr_large'
        elif params.use_bert == 2:
            path += 'xlmr'
        elif params.use_bert == 4:
            path += 'bert_large'
        elif params.use_bert == 5:
            path += 'bert_six'
        model_path = model_dir / path
        evaluate(model_path, params)





