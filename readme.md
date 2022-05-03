Code for ACL2021 paper paper 'Risk Minimization for Zero-Shot Sequence Labeling'.

### Step 1: build config file
`python train_source.py --config ner_en.config`

### Step 2: train source models(optional)
`python train_source.py --config ner_en.config`

### Step 3: construct pseudo labeled dataset
`python tools/cat_source_model.py`

### Step 4.1: Train LVM models
`python train_finetune.py --config ner_3_en.config --method lvm --table multi_logits1`

### Step 4.2: Train MRT models
`python train_finetune.py --config ner_3_en.config --method mrt --table multi_logits1`

### File structure
````
--Dataset
 --config_ner.pt
 --CoNLL
  --corpus
--RiskMin
 --.transfer_data
  --ner
  --pseudo_labeled_corpus
````