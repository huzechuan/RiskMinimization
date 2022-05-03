""" Uniformly select transfer data from n source
"""

import os
import re
import sys
import numpy as np
from pathlib import Path


def uniform_select(source, task, num_source):
    counts = [0] * (num_source)
    separator = ' ' if task in ['ner', 'wikiner', 'sem'] else '\t'
    fout = open(root / (source + '_uniform'), 'w', encoding='utf-8')
    choice = np.random.choice(range(1, num_source + 1))
    counts[choice - 1] += 1
    with open(root / source, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            line = line.strip()
            fields = re.split(separator, line)
            if line == "":
                choice = np.random.choice(range(1, num_source + 1))
                counts[choice - 1] += 1
                fout.writelines('\n')
            else:
                assert 0 < choice < len(fields) - 1, f'Wrong choice {choice}, with range of {len(fields)}, {fields}'
                token = fields[0]
                tag = str(fields[choice])
                gold = str(fields[-1])
                fout.writelines(token + ' ' + tag + ' ' + gold + '\n')
            line = f.readline()
    fout.close()
    print(counts)
    return None

if __name__=='__main__':
    root = Path('./')
    sets = os.listdir(root)
    task = 'ner'
    num_source = {'ner': 3, 'pos': 8, 'wikiner': 8, 'sem': 5}

    for s in sets:
        if '.py' in s or 'uniform' in s: continue
        uniform_select(s, task, num_source[task])
