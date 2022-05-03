import os
import re
from pathlib import Path
from typing import Set

def cat_file(files, to_path, dataset=None, task=None):
    file_lines = []
    if task in ['ner', 'wikiner', 'sem', 'entity', 'TechNews', 'Onto', 'AE'] or task.startswith('wiki'):
        seperator = ' '
    elif task =='pos':
        seperator = '\t'
    else: print('Wrong format, must be space or tab.')
    for file in files:
        with open(file / dataset, 'r') as f:
            file_lines.append(f.readlines())
    lens = list(len(f_l) for f_l in file_lines)
    assert lens.count(lens[0]) == len(lens), 'files are not from same corpus.'
    if dataset.startswith('train'): outtxt = 'train.txt'
    elif dataset.startswith('val'): outtxt = 'dev.txt'
    elif dataset.startswith('test'): outtxt = 'test.txt'
    else:
        print(f'wrong file {dataset}')
        exit(0)
    fout = open(to_path / outtxt, 'w')
    for f_tup in zip(*file_lines):
        new_line = []
        word_set = set()
        label_set = set()
        for f in f_tup:
            strs = f.strip()
            if strs == '':
                word_set.add(strs)
            else:
                fields = re.split(seperator, strs)
                new_line.append(fields[1])
                word_set.add(fields[0])
                label_set.add(fields[2])

        assert 1 == len(word_set),  'Not consistent in word'# word
        word = word_set.pop()
        if word == '':
            assert len(label_set) == 0, 'Not consistent in line, expected space but got labels.'
            fout.writelines('\n')
        else:
            new_line = seperator.join([fields[0]] + new_line + [fields[2]]) + '\n'
            assert 1 == len(label_set),  'Not consistent in label'# gold label
            fout.writelines(new_line)



    fout.close()


if __name__ == '__main__':
    # predicate label in second column
    root = '../.transfer_data'
    task = 'AE'

    # source or target may be list or str w.r.t [many -> one] or [one -> many]
    # source_langs = ['ud_english', 'ud_french', 'ud_indonesian', 'ud_hebrew',
    #                 'ud_dutch', 'ud_russian', 'ud_polish', 'ud_greek']
    # source_langs = ['ud_en', 'ud_ca', 'ud_id', 'ud_hi', 'ud_fi', 'ud_ru']
    # model_type = 'softmax_mbert'
    # source_models = ['en', 'ca', 'id', 'hi', 'fi', 'ru']
    # targets = ['ud_en', 'ud_ca', 'ud_id', 'ud_hi', 'ud_fi', 'ud_ru']
    source_models = ['en', 'de', 'nl', 'es']
    model_type = 'softmax_mbert'
    source_langs = ['conll_03_english', 'conll_03_german', 'conll_03_dutch', 'conll_03_spanish']
    targets = ['conll_03_english', 'conll_03_german', 'conll_03_dutch', 'conll_03_spanish']
    # source_models = ['bc', 'bn', 'mz', 'nw', 'tc', 'wb']
    # model_type = 'softmax_mbert'
    # source_langs = ['bc', 'bn', 'mz', 'nw', 'tc', 'wb']
    # targets = ['bc', 'bn', 'mz', 'nw', 'tc', 'wb']
    # source_models = ['en', 'es', 'nl', 'ru', 'tr']
    # model_type = 'softmax_mbert'
    # source_langs = ['en', 'es', 'nl', 'ru', 'tr']
    # targets = ['en', 'es', 'nl', 'ru', 'tr']

    # source_models = ['en', 'es', 'fr', 'ru']
    # model_type = 'softmax_xlmr'
    # source_langs = ['en', 'es', 'fr', 'ru']
    # targets = ['en', 'es', 'fr', 'ru']
    for index in range(len(targets)):
        target = targets[index]
        source_langs_copy = source_langs.copy()
        source_models_copy = source_models.copy()
        source_langs_copy.remove(source_langs_copy[index])
        source_models_copy.remove(source_models_copy[index])

        files = [Path(root) / task / lang / model_type / ('_'.join([model, target])) for lang, model in zip(source_langs_copy, source_models_copy)]
        to_path = Path(root) / task / ('_'.join(source_models_copy + ['to', target]))

        if not (os.path.exists(to_path) and os.path.isdir(to_path)):
            os.makedirs(to_path, exist_ok=True)
        for dataset in ['train_q.txt', 'val_q.txt', 'test_q.txt']:
            cat_file(files, to_path, dataset=dataset, task=task)

