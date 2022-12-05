from cProfile import label
import json
import os, importlib
import sys
from typing import Any
from transformers import AdamW,get_linear_schedule_with_warmup 
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
import argparse
from preprocess import data_conll,tools
from tqdm import tqdm

import random
from src.model import DADP
from src import span_loss
from src import metrics

from src.basic_utils import init_args, init_logger, init_seed
init_seed(seed= 42)

#global setting
import os
os.environ["TOKENIZERS_PARALLELISM"] = 'true'
device = torch.device('cuda')

def train(args,
          logger,
        src_train_data,
        src_test_data,
        tar_train_data,
        load_finetune=True,
        dp_num_label=46,
        ner_num_label=5):
    '''
    ner_num_label is useless, it can be assigned any value
    '''
    model = DADP(args,dp_num_label=dp_num_label,ner_num_label=ner_num_label,device=device).to(device)

    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)
    
    num_training_steps=len(src_train_data)*args.batch_size*args.epoch
    warmup_steps=num_training_steps*args.warmup_proportion
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=num_training_steps)

    #---loss function---
    dp_span_loss_func = span_loss.Span_loss(dp_num_label).to(device)
    span_acc = metrics.metrics_span().to(device)

    global_step=0
    best=0
    training_loss=0.0
    count=0
    lambda_weight = torch.FloatTensor([1.0]).to(device)
    beta_weight = torch.FloatTensor([0.0]).to(device)
    for epoch in range(args.epoch):
        model.train()
        model.zero_grad()
        for src_item,tar_item in tqdm(zip(src_train_data,tar_train_data),total=len(tar_train_data),unit='batches'):
            global_step+=1
            src_input_ids, src_attention_mask, src_token_type_ids = src_item["input_ids"], src_item["attention_mask"], src_item["token_type_ids"]
            src_dp_span_label, src_dp_span_mask = src_item['span_label'], src_item["span_mask"]
            src_dp_span_logits, src_ner_span_logits, src_dp_start, src_dp_end = model( 
                input_ids = src_input_ids.to(device), 
                attention_mask = src_attention_mask.to(device),
                token_type_ids = src_token_type_ids.to(device),
            )
            dp_loss = dp_span_loss_func(src_dp_span_logits, src_dp_span_label.to(device), src_dp_span_mask.to(device))

            tar_input_ids, tar_attention_mask, tar_token_type_ids = tar_item["input_ids"], tar_item["attention_mask"], tar_item["token_type_ids"]
            tar_ner_span_label, tar_ner_span_mask = tar_item['span_label'], tar_item["span_mask"]
            tar_dp_span_logits, tar_ner_span_logits, tar_dp_start, tar_dp_end = model( 
                input_ids = tar_input_ids.to(device), 
                attention_mask = tar_attention_mask.to(device),
                token_type_ids = tar_token_type_ids.to(device),
            )
            ner_loss = ner_span_loss_func(tar_ner_span_logits, tar_ner_span_label.to(device), tar_ner_span_mask.to(device))
            ner_loss=ner_loss.float().mean().type_as(ner_loss)

            mmd_loss_start = mmd_loss_func(src_dp_start, tar_dp_start, src_attention_mask, tar_attention_mask)
            mmd_loss_end = mmd_loss_func(src_dp_end, tar_dp_end, src_attention_mask, tar_attention_mask)
            

            loss = lambda_weight*(mmd_loss_start+mmd_loss_end + dp_loss) + beta_weight*ner_loss
            training_loss+=loss.item()
            count+=1
            loss.backward()
            #training_loss+=loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        model.eval()
        recall,precise,span_f1=evaluate(model=model,evaluate_data=src_test_data,span_acc=span_acc,mode='dp')
        model.train()

        logger.info('Evaluating the model...')
        logger.info('epoch %d, loss %.4f, recall %.4f, precise %.4f, span_f1 %.4f'% (epoch,training_loss/count,recall,precise,span_f1))
        training_loss=0.0
        count=0
        
        if best < span_f1:
            best=span_f1
            torch.save(model.state_dict(), f=args.checkpoints)
            logger.info('save the best model in {}'.format(args.checkpoints))   

def evaluate(model,evaluate_data,span_acc,mode='dp'):
    if model.training:
        model.eval()

    count=0
    span_f1=0
    recall=0
    precise=0

    with torch.no_grad():
        for item in evaluate_data:
            count+=1
            input_ids, attention_mask, token_type_ids = item["input_ids"], item["attention_mask"], item["token_type_ids"]
            span_label,span_mask = item['span_label'],item["span_mask"]

            dp_span_logits, ner_span_logits, dp_start, dp_end = model( 
                input_ids=input_ids.to(device), 
                attention_mask=attention_mask.to(device),
                token_type_ids=token_type_ids.to(device),
                ) 
            if mode=='ner':
                evaluate_logits=ner_span_logits
            else:
                evaluate_logits=dp_span_logits

            tmp_recall,tmp_precise,tmp_span_f1=span_acc(logits=evaluate_logits,labels=span_label.to(device),span_mask=span_mask.to(device))
            
            span_f1+=tmp_span_f1
            recall+=tmp_recall
            precise+=tmp_precise

    span_f1 = span_f1/count
    recall=recall/count
    precise=precise/count
    
    return recall,precise,span_f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp_train_path", type=str, required=True,help="train file")
    parser.add_argument("--dp_test_path", type=str, required=True,help="test file")
    parser.add_argument("--checkpoints", type=str, required=True,help="output_dir")
    parser.add_argument("--batch_size", type=int, default=32,help="batch_size")
    parser.add_argument("--lstm_hidden_size", type=int, default=512,help="lstm_hidden_size")
    parser.add_argument("--to_biaffine_size", type=int, default=128,help="to_biaffine_size")
    parser.add_argument("--max_length", type=int, default=196,help="max_length")
    parser.add_argument("--epoch", type=int, default=100,help="epoch")
    parser.add_argument("--learning_rate", type=float, default=5e-5,help="learning_rate")
    parser.add_argument("--filetune_path", type=str, default="",help="output_dir")
    parser.add_argument("--pretrained_model_path", type=str, default="bert-large-uncased-whole-word-masking",help="pretrained_model_path")
    parser.add_argument("--clip_norm", type=float, default=1,help="clip_norm")
    parser.add_argument("--warmup_proportion", type=float, default=0.08,help="warmup proportion")
    parser.add_argument("--num_workers", type=int, default=8,help='num_workers')
    parser.add_argument("--ner_train_path", type=str, default="data/W-NUT17/train_prepro_url.txt",help="train file")
    
    args = parser.parse_args()
    os.makedirs(args.checkpoints, exist_ok=True)
    logger = init_logger("main", log_file_name='{}/pretrained_on_dp_ontonotes.txt'.format(args.checkpoints))
    for k,v in args.__dict__.items():
        logger.info("{} : {}".format(str(k),str(v)))

    tokenizer=tools.get_tokenizer(bert_model_path=args.pretrained_model_path)
    dp_label2id,dp_id2label,dp_num_label=tools.load_schema_dp()
    dp_num_label+=1
    ner_label2id=tools.generate_label2id(file_path=args.ner_train_path)
    ner_label2id=tools.process_nerlabel(label2id=ner_label2id)
    tar_train_data  = data_conll.yield_data(args=args,
                                            file_path=args.ner_train_path, 
                                            tokenizer=tokenizer, 
                                            mode='ner', 
                                            label2id=ner_label2id) #ner_train -> whole span, sub span, non span 

    src_train_data  = data_conll.yield_data(args=args,
                                            file_path=args.dp_train_path, 
                                            tokenizer=tokenizer, 
                                            mode='dp',
                                            label2id=dp_label2id)

    src_test_data  = data_conll.yield_data(args=args,
                                        file_path=args.dp_test_path, 
                                        tokenizer=tokenizer, 
                                        mode='dp',
                                        label2id=dp_label2id,
                                        is_training=False)

    train(args=args,
          logger = logger,
        src_train_data=src_train_data,
        src_test_data=src_test_data,
        tar_train_data=tar_train_data,
        dp_num_label=dp_num_label,
        load_finetune=True)

if __name__=="__main__":
    main()