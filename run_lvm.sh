target=$1
gpu=$2
# name=$3

# CoNLL with logits
CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config ner_3_$target.config --method lvm --table multi_logits1 > logs/CoNLL/LVM/multi_logits1_$target.log &
echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config ner_3_$target.config --method lvm --table multi_logits1 > logs/CoNLL/LVM/multi_logits1_$target.log &"

#CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config tech_3_$target.config --method lvm --table multi_tune3 > logs/TechNews/LVM/3$target.log &
#echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config tech_3_$target.config --method lvm --table multi_tune3 > logs/TechNews/LVM/3$target.log &"

#CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config onto_5_$target.config --method lvm --table multi_tune > logs/Onto/LVM/3$target.log &
#echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config onto_5_$target.config --method lvm --table multi_tune > logs/Onto/LVM/3$target.log &"

# Onto with logits
#CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config onto_5_$target.config --method lvm --table multi_logits > logs/Onto/LVM/multi_logits_$target.log &
#echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config onto_5_$target.config --method lvm --table multi_logits > logs/Onto/LVM/multi_logits_$target.log &"

# pos with logits
#CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config pos_5_$target.config --method lvm --table multi_logits > logs/pos/LVM/multi_logits_$target.log &
#echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config pos_5_$target.config --method lvm --table multi_logits > logs/pos/LVM/multi_logits_$target.log &"

# big pos with logits
#CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config pos_5_big100_$target.config --method lvm --table big_data > logs/sem/LVM/big_data$target.log &
#echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config pos_5_big100_$target.config --method lvm --table big_data > logs/sem/LVM/multi_logits_$target.log &"

# ICBU with logits
#CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config icbu_$target.config --method lvm --table multi_logits_xlmr_base > logs/icbu/LVM/multi_logits_xlmr_base$target.log &
#echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config icbu_$target.config --method lvm --table multi_logits_xlmr_base > logs/icbu/LVM/multi_logits_xlmr_base$target.log &"

#CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config icbu_2_$target.config --method lvm --table multi_logits_2_xlmr_base > logs/icbu/LVM/multi_logits_2_xlmr_base$target.log &
#echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config icbu_2_$target.config --method lvm --table multi_logits_2_xlmr_base > logs/icbu/LVM/multi_logits_2_xlmr_base$target.log &"

#CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config AE_$target.config --method lvm --table multi_logits_xlmr_base > logs/AE/LVM/multi_logits_xlmr_base_$target.log &
#echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config AE_$target.config --method lvm --table multi_logits_xlmr_base > logs/AE/LVM/multi_logits_xlmr_base_$target.log &"

#CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config AE_$target.config --method lvm --table multi_logits_xlmr_base_2e-3 > logs/AE/LVM/multi_logits_xlmr_base_2e-3_$target.log &
#echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_finetune.py --config AE_$target.config --method lvm --table multi_logits_xlmr_base_2e-3 > logs/AE/LVM/multi_logits_xlmr_base_2e-3_$target.log &"

