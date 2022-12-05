#!/usr/bin/env bash
# -*- coding: utf-8 -*-

REPO_PATH=.
pretrained_model_path=bert-large-uncased-whole-word-masking


finetune_path=outputs/pretrained_on_dp_ontonotes/pytorch_model.bin


ner_train_path=data/W-NUT17/train_prepro_url.txt
ner_test_path=data/W-NUT17/test_prepro_url.txt
dataset_name=WNUT
num_insert_symbols=0


# ner_train_path=data/MitRest/train.txt
# ner_test_path=data/MitRest/test.txt
# dataset_name=MitRest


# ner_train_path=data/NCBI/train.txt
# ner_test_path=data/NCBI/test.txt
# dataset_name=NCBI
# num_insert_symbols=0


# ner_train_path=data/conll03/train.txt
# ner_test_path=data/conll03/test.txt
# dataset_name=conll03
# num_insert_symbols=4


max_length=196
batch_size=32
num_workers=1
epoch=100
learning_rate=5e-5
dividing_epoch=1
prev_lambda_weight=1.0
prev_beta_weight=0.0
last_lambda_weight=0.0
last_beta_weight=1.0



checkpoints=outputs/${dataset_name}/EPOCH_${epoch}_LR_${learning_rate}_BSZ_${batch_size}_${num_insert_symbols}_pretrained
echo ${dataset_name}
echo ${finetune_path}
echo ${checkpoints}

python -u ${REPO_PATH}/run_ner.py \
    --pretrained_model_path ${pretrained_model_path} \
    --finetune_path ${finetune_path} \
    --checkpoints ${checkpoints} \
    --dataset_name ${dataset_name} \
    --ner_train_path ${ner_train_path} \
    --ner_test_path ${ner_test_path} \
    --batch_size ${batch_size} \
    --max_length ${max_length} \
    --num_workers ${num_workers} \
    --epoch ${epoch} \
    --learning_rate ${learning_rate} \
    --num_insert_symbols ${num_insert_symbols} \
    --prev_lambda_weight ${prev_lambda_weight} \
    --prev_beta_weight ${prev_beta_weight} \
    --last_lambda_weight ${last_lambda_weight} \
    --last_beta_weight ${last_beta_weight} \
    --dividing_epoch ${dividing_epoch} \
