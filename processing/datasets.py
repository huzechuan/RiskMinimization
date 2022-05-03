# from torchtext.datasets import SequenceTaggingDataset
from torchtext.vocab import Vectors
from torchtext.data import Example, Dataset, Field
import torch
from typing import Dict
from pathlib import Path
import os
import re
from models.utils import log_info
import time


class BertField(Field):
    """A Label field.

    A label field is a shallow wrapper around a standard field designed to hold labels
    for a classification task. Its only use is to set the unk_token and sequential to
    `None` by default.
    """

    def __init__(self, device=None, **kwargs):
        # whichever value is set for sequential, unk_token, and is_target
        # will be overwritten
        self.vocab: Dict = dict()
        self.device = device
        super(BertField, self).__init__(**kwargs)

    def reset_vocab(self):
        self.vocab.clear()

    def build_vocab(self, Bert_model, *args, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.

        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)

        log_info('Building BERT sentence vocabulary...')
        start_time = time.time()
        sents_num = []
        prev_num = 0
        for data in sources:
            batch = 16
            sx, sx_idx = [], []
            for x in data:
                sx.append(x[1:])
                sx_idx.append(x[0])
                if len(sx) >= batch:
                    sx_bert = Bert_model.build_vocab(sx)
                    for idx, bert in zip(sx_idx, sx_bert):
                        self.vocab[idx] = bert
                    sx, sx_idx = [], []
                    if (len(self.vocab) - prev_num )% 800 == 0: log_info(x[0], dynamic=True)

            if len(sx) > 0:
                sx_bert = Bert_model.build_vocab(sx)
                for idx, bert in zip(sx_idx, sx_bert):
                    self.vocab[idx] = bert
            sents_num.append(len(self.vocab) - prev_num)
            prev_num += sents_num[-1]

        total_time = time.time() - start_time
        speed = len(self.vocab) // total_time
        log_info(f'Finished! train_set={sents_num[0]} val_set={sents_num[1]} '
                 f'test_set={sents_num[2]} {speed} sents/sec', dynamic=True)
        # self.vocab = self.vocab_cls(None, specials=['SEP', 'CLS'], **kwargs)

    def process(self, batch, device=None):
        """ Process a list of examples to create a torch.Tensor.

        Pad, numericalize, and postprocess a batch and create a tensor.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            torch.autograd.Variable: Processed object given the input
            and custom postprocessing Pipeline.
        """
        tensor = self.numericalize(batch, device=self.device)
        return tensor

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a Variable.

        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.

        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        mask = None
        words = arr
        if self.use_vocab:
            if self.sequential:
                arr = [self.vocab[ex[0]] for ex in arr]
            else:
                arr = [self.vocab.stoi[x] for x in arr]

            if self.postprocessing is not None:
                arr, lengths, mask = self.postprocessing(arr, self.include_lengths)
            var = arr.to(self.device)
            mask = mask.to(self.device)

            if self.sequential and not self.batch_first:
                # var.t_()
                pass
            if self.sequential:
                var = var.contiguous()
        else:
            arr = arr#[ex[1:] for ex in arr]
            max_len = len(max(arr, key=len)) - 1
            var = arr
            mask = torch.zeros(
                [len(arr), max_len],
                dtype=torch.bool,
                device=self.device,
            )
            lengths = []
            for s_id, sentence in enumerate(arr):
                lens = len(sentence) - 1
                mask[s_id][:lens] = torch.ones(lens, dtype=torch.bool)
                lengths.append(lens)
            words = arr

        # var = torch.tensor(arr, dtype=self.dtype, device=device)
        if self.include_lengths:
            lengths = torch.tensor(lengths, dtype=self.dtype, device=self.device)
            return var, lengths, mask, words
        return var, mask, words


class QValueField(Field):
    """A Label field.

    A label field is a shallow wrapper around a standard field designed to hold labels
    for a classification task. Its only use is to set the unk_token and sequential to
    `None` by default.
    """

    def __init__(self, device=None, **kwargs):
        # whichever value is set for sequential, unk_token, and is_target
        # will be overwritten
        self.vocab: Dict = dict()
        self.device = device
        super(QValueField, self).__init__(**kwargs)

    def reset_vocab(self, q_values, keys, lengths):
        q_values = torch.unbind(q_values.cpu(), dim=0)
        for length, key, qValue in zip(lengths, keys, q_values):
            assert self.vocab[key].size(0) == length, 'Sequence length is not match!'
            self.vocab[key] = qValue[:length, :]

    def build_vocab(self, label_size, *args, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.

        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)

        for data in sources:
            for x in data:
                q = torch.zeros((len(x) - 1, label_size), dtype=torch.float32)
                self.vocab[x[0]] = q

    def process(self, batch, device=None):
        """ Process a list of examples to create a torch.Tensor.

        Pad, numericalize, and postprocess a batch and create a tensor.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            torch.autograd.Variable: Processed object given the input
            and custom postprocessing Pipeline.
        """
        tensor = self.numericalize(batch, device=self.device)
        return tensor

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a Variable.

        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.

        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """

        if self.use_vocab:
            if self.sequential:
                keys = [ex[0] for ex in arr]
                arr = [self.vocab[ex[0]] for ex in arr]
            else:
                arr = [self.vocab.stoi[x] for x in arr]

            if self.postprocessing is not None:
                if self.include_lengths:
                    arr, lengths, mask = self.postprocessing(arr, self.include_lengths)
                else:
                    arr = self.postprocessing(arr, self.include_lengths)
        else:
            if self.dtype not in self.dtypes:
                raise ValueError(
                    "Specified Field dtype {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.dtype))
            numericalization_func = self.dtypes[self.dtype]
            # It doesn't make sense to explicitly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            if not self.sequential:
                arr = [numericalization_func(x) if isinstance(x, six.string_types)
                       else x for x in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

        # var = torch.tensor(arr, dtype=self.dtype, device=device)
        var = arr.to(self.device)
        if self.sequential and not self.batch_first:
            # var.t_()
            pass
        if self.sequential:
            var = var.contiguous()

        if self.include_lengths:
            mask = mask.to(self.device)
            lengths = torch.tensor(lengths, dtype=self.dtype, device=self.device)
            return var, keys, lengths, mask
        return var, keys


class SequenceTaggingDataset(Dataset):
    """Defines a dataset for sequence tagging. Examples in this dataset
    contain paired lists -- paired list of words and tags.

    For example, in the case of part-of-speech tagging, an example is of the
    form
    [I, love, PyTorch, .] paired with [PRON, VERB, PROPN, PUNCT]

    See torchtext/test/sequence_tagging.py on how to use this class.
    """

    @staticmethod
    def sort_key(example):
        for attr in dir(example):
            if not callable(getattr(example, attr)) and \
                    not attr.startswith("__"):
                return len(getattr(example, attr))
        return 0

    def __init__(self, path, fields, encoding="utf-8", separator="\t", **kwargs):
        examples = []
        columns = []
        if fields[0][0] is not None:
            bert = 'bert' in fields[0][0]
            qValue = 'qValue' in fields[0][0]
        else:
            bert = False
            qValue = False

        sent_id = 0
        sett = re.split('/', path)[-1].split('.')[0]
        with open(path, encoding=encoding) as input_file:
            for line in input_file:
                line = line.strip()
                if line == "":
                    if columns:
                        # if len(columns[0]) > 150:
                        #     continue
                        example = Example.fromlist(columns, fields)
                        if bert: example.bert = [f'{sett}_{sent_id}'] + example.bert
                        if qValue: example.qValue = [f'{sett}_{sent_id}'] + example.qValue
                        examples.append(example)
                        sent_id += 1
                    columns = []
                elif line.startswith('#'):
                    continue
                else:
                    for i, column in enumerate(line.split(separator)):
                        if len(columns) < i + 1:
                            columns.append([])
                        columns[i].append(column)

            if columns:
                example = Example.fromlist(columns, fields)
                if bert: example.bert = [f'{sett}_{sent_id}'] + example.bert
                if qValue: example.qValue = [f'{sett}_{sent_id}'] + example.qValue
                examples.append(example)
                sent_id += 1
        super(SequenceTaggingDataset, self).__init__(examples, fields,
                                                     **kwargs)

class SequenceData(SequenceTaggingDataset):

    # Universal Dependencies English Web Treebank.
    # Download original at http://universaldependencies.org/
    # License: http://creativecommons.org/licenses/by-sa/4.0/
    # urls = ['https://bitbucket.org/sivareddyg/public/downloads/en-ud-v2.zip']
    # dirname = 'en-ud-v2'
    # name = 'udpos'
    # def __init__(self, datasets):
    #     root = Path('.data/corpus')
    #     self.data = Path / datasets

    @classmethod
    def splits(cls, fields, root=".data/corpus/", task=None, corpora=None, separator=' ',
               train="train.txt",
               validation="dev.txt",
               test="test.txt",**kwargs):
        """Downloads and loads the Universal Dependencies Version 2 POS Tagged
        data.
        """
        cls.name = task
        cls.dirname = corpora

        return super(SequenceData, cls).splits(
            fields=fields, separator=separator, root=str(root), train=train, validation=validation,
            test=test, **kwargs)

class MyEmbeddings(Vectors):

    url_base = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/conll_{}.txt'
    root = Path('./.data/embeddings')

    def __init__(self, language="english", **kwargs):
        url = self.url_base.format(language)
        name = os.path.basename(url)
        super(MyEmbeddings, self).__init__(name, cache=self.root, url=url, **kwargs)