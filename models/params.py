from pathlib import Path
import re
import torch


class Params:
    def __init__(self, config):
        super(Params, self).__init__()
        self.config = config

        # self.tensorboard = config.getboolean('Visualization', 'tensorboard')
        # self.tensor_path = config.get('Visualization', 'tensor_path')
        self.use_transfer = config.getboolean('Task', 'transfer')
        self.task = config.get('Task', 'task')
        self.metric = config.get('Task', 'metric')

        if self.use_transfer:
            self.softem = config.getboolean('Model', 'softem')
            self.model_type = config.get('Model', 'model_type')
            self.num_decoders = config.getint('Model', 'num_decoders')
            self.dropout = config.getfloat('Model', 'dropout')
            self.use_bert = config.getint('Model', 'use_bert')
            self.berts = ['xlm-roberta-large', 'bert-base-multilingual-cased', 'xlm-roberta-base',
                          'bert-base-chinese', 'bert-large-uncased', 'google/bert_uncased_L-6_H-768_A-12']
            if self.use_bert != -1:
                self.bert = self.berts[self.use_bert]
                self.embedding = None
                self.use_char = False
            else:
                self.embedding = config.get('Model', 'word_embeddings')
                self.use_char = config.getboolean('Model', 'use_char')

            self.load_model = config.getboolean('Model', 'load_model')
            if self.load_model:
                load_model_path = config.get('Model', 'load_model_path')
                self.load_model_path = load_model_path.split(',')
                # self.static_model = torch.load(self.load_model_path)

            self.num_encoder_layers = config.getint('Encoder', 'num_layers')
            self.encoder_hidden_size = config.getint('Encoder', 'hidden_size')

            self.num_decoder_layers = config.getint('Decoder', 'num_layers')
            self.decoder_hidden_size = config.getint('Decoder', 'hidden_size')

            self.corpus = config.get('Data', 'corpus')
            self.domain = config.get('Data', 'domain')
            self.dataformat = config.get('Data', 'dataformat')
            self.result_root = config.get('Data', 'result_root')
            self.sample = config.getfloat('Data', 'sample')
            self.root = config.get('Data', 'data_root')
            self.table = config.get('Data', 'table')

            self.rounds = config.getint('Training', 'rounds')
            self.batch = config.getint('Training', 'batch')
            self.HP_lr = config.getfloat('Training', 'lr')
            self.HP_top_lr = config.getfloat('Training', 'top_lr')
            self.lr_decay = config.getfloat('Training', 'lr_decay')
            self.max_epochs = config.getint('Training', 'epochs')
            self.patience = config.getint('Training', 'patience')
            self.L2 = config.getfloat('Training', 'L2')
            self.model_path = Path('./results') / self.task / self.corpus
            self.anneal_method = 'max'
            self.mode = config.get('Training', 'mode')
            self.mu = config.get('Training', 'mu')

        else:
            self.use_bert = config.getint('Encoder_Layer', 'use_bert')
            self.berts = ['xlm-roberta-large', 'bert-base-multilingual-cased', 'xlm-roberta-base',
                          'bert-base-chinese', 'bert-large-uncased', 'google/bert_uncased_L-6_H-768_A-12']
            self.freeze = config.getboolean('Encoder_Layer', 'freeze')
            if self.use_bert != -1:
                self.bert = self.berts[self.use_bert]
                self.embedding = None
                self.use_char = False
            else:
                self.embedding = config.get('Encoder_Layer', 'word_embeddings')
                self.use_char = config.getboolean('Encoder_Layer', 'use_char')

            self.load_model = config.getboolean('Model', 'load_model')
            if self.load_model:
                load_model_path = config.get('Model', 'load_model_path')
                self.load_model_path = load_model_path.split(',')

            self.hidden_size = config.getint('Encoder_Layer', 'hidden_size')
            self.dropout = config.getfloat('Encoder_Layer', 'dropout')
            self.num_layers = config.getint('Encoder_Layer', 'num_layers')

            self.use_crf = config.getboolean('Top_Layer', 'use_crf')
            self.multilinear = config.get('Top_Layer', 'multilinear')
            self.HP_tag_dim = config.getint('Top_Layer', 'HP_tag_dim')
            self.HP_rank = config.getint('Top_Layer', 'HP_rank')
            self.HP_std = config.getfloat('Top_Layer', 'HP_std')
            self.HP_tag_scale = config.getfloat('Top_Layer', 'HP_tag_scale')

            target_corpus = config.get('Data', 'target_corpus')
            self.target_corpus = re.split(',', target_corpus)
            self.source_corpus = config.get('Data', 'source_corpus')
            self.dataformat = config.get('Data', 'dataformat')
            self.result_root = config.get('Data', 'result_root')
            self.sample = config.getfloat('Data', 'sample')
            self.source_name = config.get('Data', 'source_model_name')
            self.num_decoders = config.getint('Data', 'num_decoders')
            self.root = config.get('Data', 'data_root')
            self.corpus = config.get('Data', 'corpus')

            self.rounds = config.getint('Training', 'rounds')
            self.batch = config.getint('Training', 'batch')
            self.HP_lr = config.getfloat('Training', 'lr')
            self.lr_decay = config.getfloat('Training', 'lr_decay')
            self.max_epochs = config.getint('Training', 'epochs')
            self.patience = config.getint('Training', 'patience')
            self.L2 = config.getfloat('Training', 'L2')
            self.model_path = Path(self.result_root) / self.task / self.source_corpus

            self.anneal_method = config.get('Training', 'anneal_method')#'max'
            self.mode = config.get('Training', 'mode')
            self.save_model = config.getboolean('Training', 'save_model')


            if self.mode in 'tune':
                self.tune = {}
                tunes_int = config.items('Tune_Int')
                for t, values in tunes_int:
                    scale = list(map(int, re.split(',', values)))
                    if len(scale) > 1:
                        self.tune[t] = scale
                    elif len(scale) == 1:
                        exec(f'self.{t} = {scale[0]}')
                    else:
                        raise Exception("Valid tune parameters!")
                for t, values in config.items('Tune_Float'):
                    scale = list(map(float, re.split(',', values)))
                    if len(scale) > 1:
                        self.tune[t] = scale
                    elif len(scale) == 1:
                        exec(f'self.{t} = {scale[0]}')
                    else:
                        raise Exception("Valid tune parameters!")

                if len(self.tune) < 1:
                    self.mode = 'train'
                else:
                    assert len(
                        self.tune) == 1, f"Num of Parameters exceed constraint, expect 1, but get {len(self.tune)}"

    def set_transfer(self, config):
        # Encoder
        pass

    def reset(self, mu, top_lr, lr, max_epoch, init_mu):
        self.mu = mu
        self.HP_top_lr = top_lr
        self.HP_lr = lr
        self.max_epochs = max_epoch
        self.init_mu = init_mu

    def reset_mu(self, mu):
        self.mu = mu


class Vote(Params):
    def __init__(self, config, method=None):
        # super(Vote, self).__init__()
        self.config = config
        self.method = method

        # self.tensorboard = config.getboolean('Visualization', 'tensorboard')
        # self.tensor_path = config.get('Visualization', 'tensor_path')
        self.use_transfer = config.getboolean('Task', 'transfer')
        self.task = config.get('Task', 'task')
        self.metric = config.get('Task', 'metric')

        if not self.use_transfer:
            self.softem = config.getboolean('Model', 'softem')
            self.model_type = config.get('Model', 'model_type')
            self.num_decoders = config.getint('Model', 'num_decoders')
            self.dropout = config.getfloat('Model', 'dropout')
            self.use_bert = config.getint('Model', 'use_bert')
            self.berts = ['xlm-roberta-large', 'bert-base-multilingual-cased', 'xlm-roberta-base',
                          'bert-base-chinese']
            if self.use_bert != -1:
                self.bert = self.berts[self.use_bert]
                self.embedding = None
                self.use_char = False
            else:
                self.embedding = config.get('Model', 'word_embeddings')
                self.use_char = config.getboolean('Model', 'use_char')

            self.load_model = config.getboolean('Model', 'load_model')
            if self.load_model:
                self.load_model_path = config.get('Model', 'load_model_path')
                self.static_model = torch.load(self.load_model_path)

            self.num_encoder_layers = config.getint('Encoder', 'num_layers')
            self.encoder_hidden_size = config.getint('Encoder', 'hidden_size')

            self.num_decoder_layers = config.getint('Decoder', 'num_layers')
            self.decoder_hidden_size = config.getint('Decoder', 'hidden_size')

            self.corpus = config.get('Data', 'corpus')
            self.domain = config.get('Data', 'domain')
            self.dataformat = config.get('Data', 'dataformat')
            self.result_root = config.get('Data', 'result_root')
            self.sample = config.getfloat('Data', 'sample')
            self.root = config.get('Data', 'data_root')
            self.table = config.get('Data', 'table')

            self.rounds = config.getint('Training', 'rounds')
            self.batch = config.getint('Training', 'batch')
            self.HP_lr = config.getfloat('Training', 'lr')
            self.lr_decay = config.getfloat('Training', 'lr_decay')
            self.max_epochs = config.getint('Training', 'epochs')
            self.patience = config.getint('Training', 'patience')
            self.L2 = config.getfloat('Training', 'L2')
            self.model_path = Path('./results') / self.model_type / self.task / self.corpus
            self.anneal_method = 'max'
            self.mode = config.get('Training', 'mode')
            self.mu = config.get('Training', 'mu')
