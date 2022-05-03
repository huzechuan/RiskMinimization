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
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
          max_m_epoch: int = 1):
    # Calculate and reset QValue
    network.train()
    log_info('-' * 100)
    start_time = time.time()

    total_number_of_batches = train_iter.iterations
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

    epoch_loss = epoch_loss / (index + 1)
    print(f'')
    log_info(f'loss: {epoch_loss}.', dynamic=False)
    if params.tensorboard:
        writer.add_scalar('Loss/Train', epoch_loss, Epoch + max_m_epoch * Epoch)

    return None


def evaluate(model_path, params):
    model_data = torch.load(model_path / params.source_name / 'best_model.pt')
    Bert_model = BertEmbeddings(params.bert)
    bert: Field = BertField(device=device, use_vocab=True, eos_token='<eos>',
                            pad_token='<pad>', include_lengths=True,
                            postprocessing=bert_pad)
    label: Field = Field(pad_token='<pad>')
    fields: List = [('bert', bert), ('label', label)]
    label.vocab = model_data['label']
    eval_transfer_model = LSTM_Net(tag_map=label.vocab.stoi, embedding_length=Bert_model.embedding_length,
                                   hidden_dim=params.hidden_size * 2, rnn_layers=params.num_layers,
                                   dropout_ratio=params.dropout, use_crf=params.use_crf)
    eval_transfer_model.load_state_dict(model_data['model_state_dict'])
    if_save = True# if params.mode != 'eval' else False
    evaluator = Evaluator(params.metric, model_path, label.vocab.itos, is_transfer=False, if_save=if_save)

    eval_transfer_model.to(device)
    separator = '\t' if params.task in ['pos'] else ' '

    for target_corpus in params.target_corpus:
        log_info('-' * 100)
        log_info(f'target {target_corpus}')

        train_set, val_set, test_set = SequenceData.splits(fields=fields, task=params.task,
                                                           corpora=target_corpus, separator=separator)
        log_info(f'Corpus: "train {len(train_set)} val {len(val_set)} test {len(test_set)}"')
        train_iter_t, val_iter_t, test_iter_t = Iterator.splits((train_set, val_set, test_set),
                                                                batch_sizes=(50, 50, 50), shuffle=False)
        bert.reset_vocab()
        bert.build_vocab(Bert_model, train_set.bert, val_set.bert, test_set.bert)

        target_path = model_path / (params.source_name + '_' + target_corpus)
        if not (os.path.exists(target_path) and os.path.isdir(target_path)):
            os.makedirs(target_path, exist_ok=True)
        evaluator.reset_path(target_path)
        train_score = evaluator.cal_score(eval_transfer_model, train_iter_t, 'train', device)
        val_score = evaluator.cal_score(eval_transfer_model, val_iter_t, 'val', device)
        test_score = evaluator.cal_score(eval_transfer_model, test_iter_t, 'test', device)
        log_info(f'train_score {train_score} val_score {val_score} test_score {test_score}')

    return None


def construct_data(params: Params, device):
    # word: Field = Field(eos_token='<eos>', pad_token='<pad>', include_lengths=True)
    bert: Field = BertField(device=device, use_vocab=True, eos_token='<eos>',
                            pad_token='<pad>', include_lengths=True,
                            postprocessing=bert_pad)
    label: Field = Field(pad_token='<pad>')
    # char_nesting: Field = Field(tokenize=list, pad_token="<p>")
    # char: NestedField = NestedField(char_nesting, include_lengths=True)

    # fields: List = [(('word', 'char'), (word, char)), ('label', label)]
    fields: List = [('bert', bert), ('label', label)]
    print(params.task, params.source_corpus, params.target_corpus)
    separator = '\t' if params.task in ['pos'] else ' '
    train_set, val_set, test_set = SequenceData.splits(fields=fields, task=params.task,
                                                       corpora=params.source_corpus, separator=separator)

    # word.build_vocab(train_set.word)
    # train_word: Set = set(word.vocab.itos)
    # word.vocab = None
    # word.build_vocab(train_set.word, val_set.word, test_set.word)
    # char.build_vocab(train_set.char, val_set.char, test_set.char)
    label.build_vocab(train_set.label)
    label_dict = label.vocab
    print(label_dict.stoi)
    embeddings_tmp: List[TokenEmbeddings] = []
    emb_type = ''
    if params.use_bert == -1:
        assert (params.embedding != 'None') and (params.embedding != ' ') and (params.embedding is not None), \
            f'No such embedding: {params.embedding}\n!'
        word_embeddings = WordEmbeddings(params.embedding, train_word, word.vocab.stoi)
        embeddings_tmp.append(word_embeddings)
        emb_type = 'w'
        if params.use_char:
            char_embeddings = CharacterEmbeddings(char.vocab.stoi)
            embeddings_tmp.append(char_embeddings)
            emb_type += 'c'
    else:
        assert (params.bert != 'None') and (params.bert != ' ') and (params.bert != None), \
            f"No such BERT: {params.bert}"
        Bert_model = BertEmbeddings(params.bert)
        embeddings_tmp.append(Bert_model)
        if params.mode != 'eval':
            bert.build_vocab(Bert_model, train_set.bert, val_set.bert, test_set.bert)
        emb_type = 'b'

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings_tmp)
    sys.stdout.flush()

    # train_iter = BucketIterator.splits(
    #     (train_set), batch_size=(params.batch), sort_key=lambda x: len(x.text), shuffle=True)
    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train_set, val_set, test_set), batch_sizes=(32, 50, 50), sort_key=lambda x: len(x.bert))

    log_info('-' * 100)
    log_info('-' * 100)
    log_info(f'Corpus: "train {len(train_set)} val {len(val_set)} test {len(test_set)}"')
    log_info('-' * 100)
    log_info(f'Embedding: "{embeddings}"')
    log_info(f' - mini_batch_size: "{params.batch}"')

    return (label_dict, train_iter, val_iter, test_iter, embeddings, emb_type)


def run(params: Params, data: Tuple, device: str):
    (label_vocab, train_iter, dev_iter, test_iter, embeddings, emb_type) = data

    label_dict = label_vocab.stoi
    eval_dict = label_vocab.itos
    raw_transfer_model = LSTM_Net(tag_map=label_dict, embedding_length=embeddings.embedding_length,
                                  hidden_dim=params.hidden_size * 2, rnn_layers=params.num_layers,
                                  dropout_ratio=params.dropout, use_crf=params.use_crf)
    raw_transfer_model.rand_init()
    optimizer = optim.SGD(raw_transfer_model.parameters(), lr=params.HP_lr, weight_decay=params.L2)

    raw_transfer_model.to(device)
    if params.tensorboard:
        from collections import namedtuple
        NT = namedtuple('a', ['arg0', 'arg1'])
        batch = next(iter(train_iter))
        x = NT(batch.bert[0], batch.bert[1])
        # writer.add_graph(transfer_model, x)
        # writer.flush()

    model_dir = Path(params.model_path)
    log_info(f'Model: "{raw_transfer_model}"')
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
    track_list = list()
    epoch_list = range(0, params.max_epochs)
    patience_count = 0

    if params.mode in 'tune':
        model_path = model_dir / f'{params.HP_lr}_{params.hidden_size}_{params.HP_tag_dim}_{params.HP_rank}'
    else:
        path = ''
        if params.use_crf:
            path += 'crf_'
        else:
            path += 'softmax_'
        if params.use_bert != -1:
            path += 'mbert'
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
    evaluator = Evaluator(params.metric, source_path, eval_dict, is_transfer=False, if_save=True)
    previous_learning_rate = params.HP_lr

    scheduler: ReduceLROnPlateau = ReduceLROnPlateau(
        optimizer,
        factor=params.lr_decay,
        patience=params.patience,
        mode=params.anneal_method,
        verbose=True,
    )

    total_number_of_batches = len(train_iter)
    modulo = max(1, int(total_number_of_batches / 10))

    for epoch_idx, start in enumerate(epoch_list):
        # get new learning rate
        for group in optimizer.param_groups:
            learning_rate = group["lr"]

        # reload last best model if annealing with restarts is enabled

        # stop training if learning rate becomes too small
        if learning_rate < 0.0001:
            log_info('-' * 100)
            log_info("learning rate too small - quitting training!")
            log_info('-' * 100)
            break
        batch_time = 0
        log_info(f'learning_rate: {learning_rate:.4f}')

        train(train_iter, raw_transfer_model, optimizer, device, epoch_idx)

        val_score, _, _, _ = evaluator.cal_score(raw_transfer_model, dev_iter, 'val', device)
        if best_score < val_score:
            if params.save_model:
                torch.save({
                    'label': label_vocab,
                    'embeddings': params.bert,
                    'task': params.task,
                    'source_corpus': params.source_corpus,
                    'metric': params.metric,
                    'model_state_dict': raw_transfer_model.state_dict()
                }, source_path / 'best_model.pt')
            best_score = val_score
            test_score, _, _, _ = evaluator.cal_score(raw_transfer_model, test_iter, 'test', device)

        log_info(f'val_score {val_score} test_score {test_score}')

        if params.tensorboard:
            writer.add_scalar('F1/Validation', val_score, epoch_idx)
            writer.add_scalar('F1/Test', test_score, epoch_idx)
        scheduler.step(best_score)

    train_score, _, _, _ = evaluator.cal_score(raw_transfer_model, train_iter, 'train', device)
    log_info(f'train_score {train_score} val_score {best_score} test_score {test_score}')

    if params.mode == 'online':
        evaluate(model_path, params)
    # print(transfer_model.label_embeddings)
    if params.tensorboard:
        writer.close()
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning with ncrf or triLinear-ncrf')
    parser.add_argument('--config', default='ner_eng2eng_b_softmax.config', help='Path of .config file.')
    parser.add_argument('--cluster', default='TitanV', help='Running on which cluster, et., AI, TitanV, and P40.')
    parser.add_argument('--free', type=int, default=0, help='GPU device')
    # a = os.listdir(Path('../.cache'))
    args = parser.parse_args()
    config_name = args.config
    cluster = args.cluster
    # torch.cuda.set_device(0)
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
    if params.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=Path('./Visualization'))
    else:
        writer = None

    # 2. Construct data and embeddings.
    data = construct_data(params, device)
    if params.mode in ['train', 'online']:
        # Each settings should run n rounds to average performance.
        for round_i in range(params.rounds):
            run(params, data, device)
    elif params.mode in ['eval']:
        model_dir = Path(params.model_path)
        path = ''
        if params.use_crf:
            path += 'crf_'
        else:
            path += 'softmax_'
        if params.use_bert != -1:
            path += 'mbert'
        model_path = model_dir / path
        evaluate(model_path, params)





