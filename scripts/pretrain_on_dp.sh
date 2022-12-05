#!/usr/bin/env bash
# -*- coding: utf-8 -*-

REPO_PATH=.
pretrained_model_path=bert-large-uncased-whole-word-masking

checkpoints=outputs/pretrained_on_dp_ontonotes/
dp_train_path=data/ontonotes5/train.gold.conllu 
dp_test_path=data/ontonotes5/test.gold.conllu
num_workers=1

max_length=256
batch_size=32

python -u ${REPO_PATH}/pretrain_on_dp.py \
    --pretrained_model_path ${pretrained_model_path} \
    --checkpoints ${checkpoints} \
    --dp_train_path ${dp_train_path} \
    --dp_test_path ${dp_test_path} \
    --batch_size ${batch_size} \
    --max_length ${max_length} \
    --num_workers ${num_workers} \
