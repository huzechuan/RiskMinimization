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

    label.build_vocab(train_set.label)
    torch.save({
        f'{params.corpus}':
            {'label_dic': label.vocab}
    }, Path(params.root) / f'config_{params.task}.pt')
    data_config = torch.load(Path(params.root) / f'config_{params.task}.pt')
    tmp = data_config[f'{params.corpus}']['label_dic']
    label.build_vocab(train_set.label)
    log_info(label.vocab.stoi)
    tmp.extend(label.vocab)
    for target_corpus in params.target_corpus:
        train_set, val_set, test_set = SequenceData.splits(fields=fields, root=params.root, task='',
                                                           corpora=target_corpus, separator=separator)
        label.build_vocab(train_set.label)
        log_info(f'{target_corpus} {label.vocab.itos}')
        tmp.extend(label.vocab)

    torch.save({
        f'{params.corpus}':
            {'label_dic': tmp}
    }, Path(params.root) / f'config_{params.task}.pt')
    print(tmp.stoi)


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