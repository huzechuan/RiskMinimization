target=$1
gpu=$2
# name=$3

# CoNLL with logits
CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config ner_3_$target.config --method mrt --table multi_logits1 > logs/CoNLL/MRT/multi_logits1_$target.log &
echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config ner_3_$target.config --method mrt --table multi_logits1 > logs/CoNLL/MRT/multi_logits1_$target.log &"

#CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config tech_3_$target.config --method mrt --table multi_tune3 > logs/TechNews/MRT/$target.log &
#echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config tech_3_$target.config --method mrt --table multi_tune3 > logs/TechNews/MRT/$target.log &"

#CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config onto_5_$target.config --method mrt --table multi_tune > logs/Onto/MRT/3$target.log &
#echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config onto_5_$target.config --method mrt --table multi_tune > logs/Onto/MRT/3$target.log &"

# Onto with logits
#CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config onto_5_$target.config --method mrt --table multi_logits > logs/Onto/MRT/multi_logits_$target.log &
#echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config onto_5_$target.config --method mrt --table multi_logits > logs/Onto/MRT/multi_logits_$target.log &"

# ICBU with logits
#CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config icbu_$target.config --method mrt --table multi_logits > logs/icbu/MRT/multi_logits_$target.log &
#echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config icbu_$target.config --method mrt --table multi_logits > logs/icbu/MRT/multi_logits_$target.log &"

# ICBU with logits
#CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config icbu_$target.config --method mrt --table multi_logits_xlmr_base > logs/icbu/MRT/multi_logits_xlmr_base$target.log &
#echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config icbu_$target.config --method mrt --table multi_logits_xlmr_base > logs/icbu/MRT/multi_logits_xlmr_base$target.log &"

#CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config icbu_2_$target.config --method mrt --table multi_logits_2_xlmr_base > logs/icbu/MRT/multi_logits_2_xlmr_base$target.log &
#echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config icbu_2_$target.config --method mrt --table multi_logits_2_xlmr_base > logs/icbu/MRT/multi_logits_2_xlmr_base$target.log &"

#CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config AE_$target.config --method mrt --table multi_logits_xlmr_base > logs/AE/MRT/multi_logits_xlmr_base_$target.log &
#echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config AE_$target.config --method mrt --table multi_logits_xlmr_base > logs/AE/MRT/multi_logits_xlmr_base_$target.log &"

