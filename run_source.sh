target=$1
gpu=$2
# name=$3

# random sample
CUDA_VISIBLE_DEVICES=$gpu nohup python train_source.py --config ner_$target.config > logs/CoNLL/source/$target.log &
echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_source.py --config ner_$target.config > logs/CoNLL/source/$target.log &"

#CUDA_VISIBLE_DEVICES=$gpu nohup python train_source.py --config onto_$target.config > logs/Onto/source/$target.log &
#echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_source.py --config onto_$target.config > logs/Onto/source/$target.log &"

#CUDA_VISIBLE_DEVICES=$gpu nohup python train_source.py --config pos_$target.config > logs/pos/source/1$target.log &
#echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_source.py --config pos_$target.config > logs/pos/source/1$target.log &"

#CUDA_VISIBLE_DEVICES=$gpu nohup python train_source.py --config sem_$target.config > logs/sem/source/$target.log &
#echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_source.py --config sem_$target.config > logs/sem/source/$target.log &"

#CUDA_VISIBLE_DEVICES=$gpu nohup python train_source.py --config icbu_$target.config > logs/icbu/source/xlmr_base_$target.log &
#echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_source.py --config icbu_$target.config > logs/icbu/source/xlmr_base_$target.log &"

#CUDA_VISIBLE_DEVICES=$gpu nohup python train_source.py --config ner_latent_$target.config > logs/ner_latent/source/bert_large_$target.log &
#echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_source.py --config ner_latent_$target.config > logs/ner_latent/source/bert_large_$target.log &"

#CUDA_VISIBLE_DEVICES=$gpu nohup python train_source.py --config AE_$target.config > logs/AE/source/xlmr_base_$target.log &
#echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_source.py --config AE_$target.config > logs/AE/source/xlmr_base_$target.log &"
