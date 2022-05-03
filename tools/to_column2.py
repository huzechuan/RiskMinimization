from pathlib import Path
import argparse
import re
import os

def convert_ud_2c(corpus, prefix):
    # prefix = 'id_gsd'
    sets = ['train', 'dev', 'test']
    label_set = set()
    to_path = to_root / corpus
    if not (os.path.exists(to_path) and os.path.isdir(to_path)):
        os.makedirs(to_path, exist_ok=True)
    for s in sets:
        data = prefix + '-ud-' + s + '.conllu'
        fout = open(to_root / corpus / (s + '.txt'), 'w')
        with open(root / corpus / data, 'r', encoding="utf-8") as file:
            line = file.readline()
            position = 0
            while line:
                line = line.strip()
                fields = re.split("\t+", line)
                if line == "":
                    fout.writelines('\n')
                elif line.startswith("#"):
                    line = file.readline()
                    continue
                elif "." in fields[0]:
                    line = file.readline()
                    continue
                elif "-" in fields[0]:
                    line = file.readline()
                    continue
                else:
                    token = fields[1]
                    tag = str(fields[3])
                    if s == 'train':
                        label_set.add(tag)
                    fout.writelines(token + '\t' + tag + '\n')
                line = file.readline()
        if s == 'train':
            if len(label_set) > len(gold_set): print('label_set > gold_set!')
            elif len(label_set) == len(gold_set):
                if label_set != gold_set: print('same size, but no same items!')
                else: print('totally same label set!')
            else:
                union_set = label_set | (gold_set - label_set)
                if union_set != gold_set: print('same size, but no same items!')
                for l in (gold_set - label_set):
                    fout.writelines(l.lower() + '\t' + l + '\n')
                    print(f'add {l} ')
            print(gold_set-label_set)
        fout.close()
    return None

if __name__=='__main__':
    gold_set = set(['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ',
                    'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
                    'SCONJ', 'SYM', 'VERB', 'X'])
    prefixs = ['de_gsd', 'he_htb', 'fa_seraji', 'es_gsd', 'nl_alpino',
               'ru_syntagrus', 'sk_snk', 'fi_tdt', 'ca_ancora', 'hi_hdtb',
               'et_edt', 'pl_lfg', 'ar_padt', 'la_proiel', 'el_gdt',
               'sv_talbanken', 'ga_idt']
    corpus = ['ud_german', 'ud_hebrew', 'ud_persian', 'ud_spanish', 'ud_dutch',
              'ud_russian', 'ud_slovak', 'ud_finnish', 'ud_catalan', 'ud_hindi',
              'ud_estonian', 'ud_polish', 'ud_arabic', 'ud_latin', 'ud_greek',
              'ud_swedish', 'ud_irish']
    parser = argparse.ArgumentParser(description='Convert UD format to 2 column.')
    parser.add_argument('--corpus', default=None, help='Path of data file.')
    parser.add_argument('--prefix', default=None, help='Path of data file.')
    args = parser.parse_args()
    root = Path('../Tri_project/.TL/datasets')
    to_root = Path('./.data/corpus/pos')
    if args.corpus is None:
        for corpora, prefix in zip(corpus, prefixs):
            convert_ud_2c(corpora, prefix)
    else:
        convert_ud_2c(args.corpus, args.prefix)

