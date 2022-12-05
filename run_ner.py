from cProfile import label
from email.policy import strict
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
import random
from src.model import DADP
from src import span_loss, mmd_loss, metrics

from src.basic_utils import init_args, init_logger, init_seed


from tqdm import tqdm

init_seed(seed= 42)

#global setting
import os
os.environ["TOKENIZERS_PARALLELISM"] = 'true'
device = torch.device('cuda')


def train_without_mmd(args,
          logger, 
        tar_train_data,
        tar_test_data,
        ner_label2id,
        tokenizer,
        load_finetune=True,
        dp_num_label=46):
    '''
    Training ner task without MMD
    '''
    saved_model_path = os.path.join(args.checkpoints, "pytorch_model.bin")
    ner_num_label=len(ner_label2id)
    model = DADP(args,dp_num_label=dp_num_label,ner_num_label=ner_num_label,device=device).to(device)

    if load_finetune and os.path.exists(args.finetune_path):
        #args.filetune_path means the checkpoints of pretrained model on PTB corpus
        logger.info("Loading pretrained dp model on PTB from {}".format(args.finetune_path))
        model_state_dict=torch.load(args.finetune_path,map_location='cpu')
        a,_,b=model_state_dict['ner_biaffne_layer.U'].size()
        model_state_dict['ner_biaffne_layer.U']=torch.FloatTensor(torch.randn((a,ner_num_label,b)))
        load_messages=model.load_state_dict(model_state_dict,strict=False)
        logger.info(str(load_messages))
        #can not load ner_biaffine_layer

    model.to(device)
    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)
    
    num_training_steps=len(tar_train_data)*args.batch_size*args.epoch
    warmup_steps=num_training_steps*args.warmup_proportion
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=num_training_steps)

    #---loss function---
    ner_span_loss_func = span_loss.Span_loss(ner_num_label,class_weight=[1]+[4]*(ner_num_label-1)).to(device)
    
    global_step=0
    best=0
    training_loss=0.0
    count=0
    for epoch in range(args.epoch):
        model.train()
        model.zero_grad()
        for tar_item in tqdm(tar_train_data,total=len(tar_train_data),unit='batches'):
            global_step+=1
            tar_input_ids, tar_attention_mask, tar_token_type_ids = tar_item["input_ids"], tar_item["attention_mask"], tar_item["token_type_ids"]
            tar_ner_span_label, tar_ner_span_mask = tar_item['span_label'], tar_item["span_mask"]
            tar_dp_span_logits, tar_ner_span_logits, tar_dp_start, tar_dp_end = model( 
                input_ids = tar_input_ids.to(device), 
                attention_mask = tar_attention_mask.to(device),
                token_type_ids = tar_token_type_ids.to(device),
            )
            ner_loss = ner_span_loss_func(tar_ner_span_logits, tar_ner_span_label.to(device), tar_ner_span_mask.to(device))
            loss=ner_loss.float().mean().type_as(ner_loss)
            training_loss+=loss.item()
            count+=1
            loss.backward()
            #training_loss+=loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        model.eval()
        recall,precise,span_f1=evaluate_ner(args=args,
                                            logger=logger,
                                            model=model,
                                            label2id=ner_label2id,
                                            tokenizer=tokenizer)
        model.train()

        logger.info('Evaluating the model...')
        logger.info('epoch %d, loss %.4f, recall %.4f, precise %.4f, span_f1 %.4f'% (epoch,training_loss/count,recall,precise,span_f1))
        training_loss=0.0
        count=0
        
        if best < span_f1:
            best=span_f1
            torch.save(model.state_dict(), f=saved_model_path)
            logger.info('save the best model in {}'.format(saved_model_path))



def train(args,
          logger, 
        tar_train_data,
        tar_test_data,
        ner_label2id,
        src_train_data,
        tokenizer,
        load_finetune=True,
        dp_num_label=46):
    
    saved_model_path = os.path.join(args.checkpoints, "pytorch_model.bin")
    ner_num_label=len(ner_label2id)
    model = DADP(args,dp_num_label=dp_num_label,ner_num_label=ner_num_label,device=device).to(device)

    if load_finetune:
        logger.info("Loading pretrained dp model from {}".format(args.finetune_path))
        model_state_dict=torch.load(args.finetune_path,map_location='cpu')
        a,_,b=model_state_dict['ner_biaffne_layer.U'].size()
        model_state_dict['ner_biaffne_layer.U']=torch.FloatTensor(torch.randn((a,ner_num_label,b)))
        load_messages=model.load_state_dict(model_state_dict,strict=False)
        logger.info(str(load_messages))
        #can not load ner_biaffine_layer

    model.to(device)
    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)
    
    num_training_steps=len(tar_train_data)*args.batch_size*args.epoch
    warmup_steps=num_training_steps*args.warmup_proportion
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=num_training_steps)

    #---loss function---
    ner_span_loss_func = span_loss.Span_loss(ner_num_label,class_weight=[1]+[4]*(ner_num_label-1)).to(device)
    dp_span_loss_func = span_loss.Span_loss(dp_num_label).to(device)
    mmd_loss_func = mmd_loss.MMD_loss().to(device)
    span_acc = metrics.metrics_span().to(device)
    
    global_step=0
    best=0
    training_ner_loss=0.0
    training_dp_loss=0.0
    count=0
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
            dp_loss=dp_loss.float().mean().type_as(dp_loss)
            
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
            
            if epoch < args.dividing_epoch:
                lambda_weight = torch.FloatTensor([args.prev_lambda_weight])
                beta_weight = torch.FloatTensor([args.prev_beta_weight])
            else:
                lambda_weight = torch.FloatTensor([args.last_lambda_weight])
                beta_weight = torch.FloatTensor([args.last_beta_weight])
            
            lambda_weight = lambda_weight.to("cuda")
            beta_weight = beta_weight.to("cuda")
            loss=lambda_weight*(mmd_loss_start+mmd_loss_end + dp_loss) + beta_weight*ner_loss
            training_ner_loss+=ner_loss.item()
            training_dp_loss+= dp_loss.item() 
            count+=1
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


        model.eval()
        recall,precise,span_f1=evaluate_ner(args=args,
                                            logger=logger,
                                            model=model,
                                            label2id=ner_label2id,
                                            tokenizer=tokenizer)
        model.train()

        logger.info('Evaluating the model...')
        logger.info('epoch %d, lambda_weight %.2f, beta_weight %.2f, loss(ner) %.4f, loss(dp) %.4f, recall %.4f, precise %.4f, span_f1 %.4f'% (epoch, lambda_weight.item(), beta_weight.item(), training_ner_loss/count,training_dp_loss/count,recall,precise,span_f1))
        training_ner_loss=0.0
        training_dp_loss=0.0
        count=0
        
        if best < span_f1:
            best=span_f1
            torch.save(model.state_dict(), f=saved_model_path)
            logger.info('save the best model in {}'.format(saved_model_path))



def evaluate_ner(args,logger, model,label2id,tokenizer):
    id2label={k:v for v,k in label2id.items()}
    ner_num_label=len(label2id)
    sentences,entities=data_conll.load_ner_data(args.ner_test_path)
    print(len(sentences),len(entities))
    print(sentences[0])
    print(entities[0])

    examples=[]
    for sentence,entity in tqdm(zip(sentences,entities),total=len(sentences),unit='sentence'):
        assert len(sentence.split(' '))==len(entity)
        example={'sentence':sentence.split(' ')}#,'entity':entity}
        inputs=tokenizer(text=sentence)
        input_ids,attention_mask,token_type_ids = inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']
        #example['input_ids']=input_ids
        #example['attention_mask']=attention_mask
        example['wordpiece_sentence']=tokenizer.convert_ids_to_tokens(input_ids)
        span_label,ner_relation=data_conll.get_span_label(sentence,tokenizer,attention_mask,relation=entity,label2id=label2id)
        #example['span_label']=span_label
        example['ner_relation']=ner_relation
        piece_length=len(attention_mask)
        example['piece_length']=piece_length
        with torch.no_grad():
            input_ids=torch.LongTensor([input_ids])
            attention_mask=torch.LongTensor([attention_mask])
            token_type_ids=torch.LongTensor([token_type_ids])
            dp_span_logits, ner_span_logits, dp_start, dp_end = model( 
                input_ids=input_ids.to(device), 
                attention_mask=attention_mask.to(device),
                token_type_ids=token_type_ids.to(device),
                ) 
        #ner_span_logits=ner_span_logits[0]
        ner_span_logits=torch.nn.functional.softmax(ner_span_logits[0],dim=-1)
        assert ner_span_logits.size()==(piece_length,piece_length,ner_num_label)
        predict_ids=torch.argmax(ner_span_logits,dim=-1)
        assert predict_ids.size()==(piece_length,piece_length)
        predict_ids=predict_ids.cpu().tolist()
        example['predict_span']=[]
        tmp_records={}
        for start_id in range(1,piece_length-1):
            for end_id in range(start_id,piece_length-1):
                if predict_ids[start_id][end_id]!=0:
                    example['predict_span'].append((start_id,
                                                    end_id,
                                                    id2label[predict_ids[start_id][end_id]],
                                                    float(ner_span_logits[start_id,end_id,predict_ids[start_id][end_id]])))
        
        examples.append(example)


    number_span_right=0
    number_span_wrong=0
    number_span_totoal=0

    for example in examples:
        predict_span=example['predict_span']
        ner_relation=example['ner_relation']
        new_predict_span = []
        
        #ruling    
        predict_span = sorted(predict_span, key=lambda x:x[3], reverse=True)
        pos=set()
        for s, e, c, score in predict_span:
            #if intersection
            is_set = False
            for (ds,de) in pos:
                if s <= de and s >= ds:
                    is_set = True
                elif e >= ds and e <= de:
                    is_set = True
            if not is_set:
                pos.add((s,e))
                new_predict_span.append([s,e,c])    
            
        #recall
        number_span_totoal += len(ner_relation)
        
        #precision
        for each in new_predict_span:
            if each in ner_relation:
                number_span_right += 1
            else:
                number_span_wrong += 1

    recall=number_span_right/number_span_totoal
    precision=number_span_right/(number_span_right+number_span_wrong)
    f1=2*precision*recall/(precision+recall) if (precision+recall != 0) else 0
    #logger.info('recall : {}, precision : {}, f1 : {}'.format(recall,precision,f1))
    return recall,precision,f1

def main(args, logger):

    tokenizer=tools.get_tokenizer(bert_model_path=args.pretrained_model_path)
    ner_label2id=tools.generate_label2id(file_path=args.ner_train_path)
    ner_label2id=tools.process_nerlabel(label2id=ner_label2id)
    logger.info("Ner label2id : {}".format(json.dumps(ner_label2id)))

    tar_train_data  = data_conll.yield_data(args=args,
                                            file_path=args.ner_train_path, 
                                            tokenizer=tokenizer, 
                                            mode='ner', 
                                            label2id=ner_label2id) #ner_train -> whole span, sub span, non span 
    tar_test_data  = data_conll.yield_data(args=args,
                                            file_path=args.ner_test_path, 
                                            tokenizer=tokenizer, 
                                            mode='ner', 
                                            label2id=ner_label2id, 
                                            is_training=False) #ner_test -> whole span, non span

    dp_label2id,dp_id2label,dp_num_label=tools.load_schema_dp()
    dp_num_label+=1
    
    src_train_data  = data_conll.yield_data(args, args.dp_train_path, tokenizer, 'dp',label2id=dp_label2id,limit=len(tar_train_data)*args.batch_size)
    assert len(src_train_data)==len(tar_train_data)
    train(args=args, logger=logger,
        tar_train_data=tar_train_data,
        tar_test_data=tar_test_data,
        ner_label2id=ner_label2id,
        src_train_data=src_train_data,
        tokenizer=tokenizer,
        load_finetune=args.load_finetune,
        dp_num_label=dp_num_label)

#     train_without_mmd(args=args, logger=logger,
#         tar_train_data=tar_train_data,
#         tar_test_data=tar_test_data,
#         ner_label2id=ner_label2id,
#         tokenizer=tokenizer,
#         load_finetune=args.load_finetune,
#         dp_num_label=dp_num_label)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp_train_path", type=str, default="data/ontonotes5/train.gold.conllu",help="train file")
    parser.add_argument("--ner_train_path", type=str, default="data/W-NUT17/train_prepro_url.txt",help="train file")
    parser.add_argument("--ner_test_path", type=str, default="data/W-NUT17/test_prepro_url.txt",help="test file")
    parser.add_argument("--checkpoints", type=str, default="W-NUT/pytorch_model.bin",help="output_dir")
    parser.add_argument("--batch_size", type=int, default=32,help="batch_size")
    parser.add_argument("--lstm_hidden_size", type=int, default=512,help="lstm_hidden_size")
    parser.add_argument("--to_biaffine_size", type=int, default=128,help="to_biaffine_size")
    parser.add_argument("--max_length", type=int, default=196,help="max_length")
    parser.add_argument("--epoch", type=int, default=100,help="epoch")
    parser.add_argument("--learning_rate", type=float, default=5e-5,help="learning_rate")
    parser.add_argument("--finetune_path", type=str, default="pretrain_on_dp/pytorch_model.bin",help="output_dir")
    parser.add_argument("--pretrained_model_path", type=str, default="bert-large-uncased-whole-word-masking",help="pretrained_model_path")
    parser.add_argument("--clip_norm", type=float, default=1,help="clip_norm")
    parser.add_argument("--warmup_proportion", type=float, default=0.08,help="warmup proportion")
    parser.add_argument("--num_workers", type=int, default=8,help='num_workers')
    parser.add_argument("--dataset_name", type = str, required=True, help="dataset name")
    parser.add_argument("--num_insert_symbols", type=int, default=0)
    parser.add_argument("--load_finetune", action="store_true")
    parser.add_argument("--dividing_epoch", type=int, default=0)
    parser.add_argument("--prev_lambda_weight", type=float, default=1)
    parser.add_argument("--prev_beta_weight", type=float, default=0.1)
    parser.add_argument("--last_lambda_weight", type=float, default=0.1)
    parser.add_argument("--last_beta_weight", type=float, default=1)

    args = parser.parse_args()
    os.makedirs(args.checkpoints, exist_ok=True)
    
    logger = init_logger("main", log_file_name='{}/ner-{}-log.txt'.format(args.checkpoints, args.dataset_name))
    for k,v in args.__dict__.items():
        logger.info("{} : {}".format(str(k),str(v)))
        
    try:
        main(args, logger)
    except Exception as e:
        print(e)
        logger.exception(e)
