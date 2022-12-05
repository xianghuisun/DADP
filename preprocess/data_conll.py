from math import log
import os
import sys
from typing import Text
sys.path.append('./')

import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import unicodedata, re
import random
from preprocess import tools
from tqdm import tqdm
import csv
from itertools import compress

import logging
logger=logging.getLogger('main.data_conll')



def load_conll03_data(file_path):
    '''
    [['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'], ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']]
    [['SOCCER', '-', 'JAPAN', 'GET', 'LUCKY', 'WIN', ',', 'CHINA', 'IN', 'SURPRISE', 'DEFEAT', '.'], ['O', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'B-PER', 'O', 'O', 'O', 'O']]
    '''
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter = ' ')
        for row in reader:
            data.append([row])


    sentence = []
    tags = []
    result=[]
    for row in data:
        # extract first element of list.
        row = row[0]
        # TO DO: move to data reader.
        if len(row) > 0 and row[0] != '-DOCSTART-':
            sentence.append(row[0])
            tags.append(row[-1])        
        if len(row) == 0 and len(sentence) > 0:
            # clean up sentence/tags.
            # remove white spaces.
            selector = [word != ' ' for word in sentence]
            sentence = list(compress(sentence, selector))
            tags = list(compress(tags, selector))
            # append if sentence length is still greater than zero..
            if len(sentence) > 0:
                result.append([sentence,tags])
            sentence = []
            tags = []
            
    return result


def add_noise(data_examples, num_insert_symbols, file_to_write):
    '''
    [['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'], ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']]
    [['SOCCER', '-', 'JAPAN', 'GET', 'LUCKY', 'WIN', ',', 'CHINA', 'IN', 'SURPRISE', 'DEFEAT', '.'], ['O', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'B-PER', 'O', 'O', 'O', 'O']]
    '''
    f= open(file_to_write, 'w')
    insert_symbols = ['.', '?', '!', '@', '#', '~', '[', ']', '&', ',']
    for i in range(len(data_examples)):
        word_list, tag_list = data_examples[i]
        assert len(word_list) == len(tag_list)
        for _ in range(num_insert_symbols):
            insert_pos = random.randint(0, len(word_list)-1)
            insert_symbol = insert_symbols[random.randint(0, len(insert_symbols)-1)]
            if tag_list[insert_pos] == 'O':
                tag_list.insert(insert_pos, 'O')
                word_list.insert(insert_pos, insert_symbol)
        #result.append([word_list, tag_list])
        for word, tag in zip(word_list, tag_list):
            f.write(word+' '+tag+'\n')
        f.write('\n')
        
    f.close()
    
    
def load_ner_data(file_path):
    logger.info("Loading data from file path %s" % file_path)
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    
    sentences = []
    relations = []
    text = []
    relation = []
    idx=1
    for line in lines:
        if line == '\n' or line=='' or line.split() == []:
            #print(len(text))
            sentence = (' ').join(text)
            if relation!=[]:
                sentences.append(sentence)
                relations.append(relation)
            #print(sentence)
            #print(relation)
            text = []
            relation = []
            idx=1
            continue
        line = line.split()

        word, rel = line[0], line[-1].strip()
        #text.append(tokenizer.tokenize(word)[0])
        text.append(word)
        relation.append((int(idx),int(idx),rel))
        idx+=1

    return sentences,relations
    

def get_span_label(sentence,tokenizer,attention_mask,relation,label2id):
    assert len(attention_mask)==sum(attention_mask)
    span_label = [0 for i in range(len(attention_mask))]#label2id['O']=0
    span_label = [span_label for i in range(len(attention_mask))]
    span_label = np.array(span_label)
    ner_relation = []
    sentence=sentence.split(' ')
    assert len(sentence)==len(relation)

    new_relation=[]
    idx=1
    for i in range(len(sentence)):
        _,_,tag=relation[i]
        wordpiece=tokenizer.tokenize(sentence[i])
        if len(wordpiece)==1:
            new_relation.append((idx,idx,tag))
            idx+=1
        else:
            for j in range(len(wordpiece)):
                cur_tag=tag
                if j>0:
                    cur_tag='I'+tag[1:] if tag!='O' else tag#process I
                new_relation.append((idx,idx,cur_tag))
                idx+=1

    relation=new_relation

    ner_relation = []
    start_idx = 0
    end_idx = 0
    pre_label = 'O'
    #relabelling
    ent_tag='O'
    relation.append((relation[-1][0]+1,relation[-1][0]+1,'O'))
    for i, (idx,_,cur_label)in enumerate(relation):

        if cur_label[0]=='O':
            if pre_label[0]!='O' and pre_label[0]!='S':
                ner_relation.append((start_idx,idx-1,ent_tag))
                start_idx=idx

        if cur_label[0]=='B':
            if pre_label[0]=='O' or pre_label[0]=='S':
                start_idx=idx
                ent_tag=cur_label[2:]
            if pre_label[0]=='I' or pre_label[0]=='E':
                ner_relation.append((start_idx,idx-1,ent_tag))
                start_idx=idx
                ent_tag=cur_label[2:]

        pre_label=cur_label

    #print(ner_relation)
    for start_idx, end_idx, rel in ner_relation:
        span_label[start_idx, end_idx] = label2id[rel]
        
    return span_label,ner_relation

def load_dp_data(file_path, tokenizer):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    
    sentences = []
    relations = []
    text = []
    relation = []
    for line in lines:
        if line == '\n' or line=='':
            #print(len(text))
            sentence = (' ').join(text)
            if relation!=[]:
                sentences.append(sentence)
                relations.append(relation)
            #print(sentence)
            #print(relation)
            text = []
            relation = []
            continue
        line = line.strip().split()
        
        idx, word, to_idx, rel = line[0], line[1], line[6], line[7]
        text.append(tokenizer.tokenize(word)[0])
        relation.append((int(idx),int(to_idx),rel))

    logger.info("File {} has {} sentences!".format(file_path,len(sentences)))
    all_sen_lengths=[len(sen.split()) for sen in sentences]
    logger.info("Average sentence length : {}".format(sum(all_sen_lengths)/len(all_sen_lengths)))
    return sentences, relations

#encode_sent is input_ids, span_label.shape==(max_length,max_length)
#span_mask.shape==(max_length,max_length)

def get_span_mask_label(args, sentence,tokenizer,attention_mask,relation,label2id,mode):
    zero = [0 for i in range(args.max_length)]
    span_mask=[ attention_mask for i in range(sum(attention_mask))]
    span_mask.extend([ zero for i in range(sum(attention_mask),args.max_length)])
    #span_mask=np.triu(np.array(span_mask)).tolist()

    span_label = [0 for i in range(args.max_length)]#label2id['O']=0
    span_label = [span_label for i in range(args.max_length)]
    span_label = np.array(span_label)
    ner_relation = []
    sentence=sentence.split(' ')
    assert len(sentence)==len(relation)

    if mode!='dp':
        assert mode=='ner'
        new_relation=[]
        idx=1
        for i in range(len(sentence)):
            _,_,tag=relation[i]
            wordpiece=tokenizer.tokenize(sentence[i])
            if mode=='dp':
                assert len(wordpiece)==1

            if len(wordpiece)==1:
                new_relation.append((idx,idx,tag))
                idx+=1
            else:
                for j in range(len(wordpiece)):
                    cur_tag=tag
                    if j>0:
                        # if tag=='O':
                        #     cur_tag=tag
                        # elif tag.split('-')>=3:
                        #     #B-creative-work
                        #     cur_tag=tag[0].replace('I')
                        cur_tag='I'+tag[1:] if tag!='O' else tag
                    new_relation.append((idx,idx,cur_tag))
                    idx+=1
                    if idx==args.max_length-2:
                        break
            if idx==args.max_length-2:
                break
        relation=new_relation

    if mode == 'dp':            
        for start_idx, end_idx, rel in relation:
            #print(start_idx, end_idx, rel)
            span_label[start_idx, end_idx] = label2id[rel]
    elif mode == 'ner':
        ner_relation = []
        start_idx = 0
        end_idx = 0
        pre_label = 'O'
        #relabelling
        ent_tag='O'
        relation.append((relation[-1][0]+1,relation[-1][0]+1,'O'))
        for i, (idx,_,cur_label)in enumerate(relation):

            if cur_label[0]=='O':
                if pre_label[0]!='O' and pre_label[0]!='S':
                    ner_relation.append((start_idx,idx-1,ent_tag))
                    start_idx=idx

            if cur_label[0]=='B':
                if pre_label[0]=='O' or pre_label[0]=='S':
                    start_idx=idx
                    ent_tag=cur_label[2:]
                if pre_label[0]=='I' or pre_label[0]=='E':
                    ner_relation.append((start_idx,idx-1,ent_tag))
                    start_idx=idx
                    ent_tag=cur_label[2:]

            pre_label=cur_label

        for start_idx, end_idx, rel in ner_relation:
            #wordpiece input
            span_label[start_idx, end_idx] = label2id[rel]
    return span_mask,span_label,ner_relation


def data_pre(args, file_path, tokenizer, mode, label2id, batch_size=128):
    if mode.lower()=='ner':
        if args.dataset_name == 'conll03':
            data_examples = load_conll03_data(file_path = file_path)
            file_to_write = os.path.join('/'.join(file_path.split('/')[:-1]), "noise_{}".format(file_path.split('/')[-1]))
            logger.info("Writing noise data to {}".format(file_to_write))
            add_noise(data_examples, args.num_insert_symbols, file_to_write)
            file_path = file_to_write
            
        sentences, relations = load_ner_data(file_path=file_path)
    else:
        sentences, relations = load_dp_data(file_path=file_path,tokenizer=tokenizer)
    data = []
    #print(len(sentences),mode)
    for _ in range(10):
        idx = random.randint(0, len(sentences)-1)
        
        for word_, tag_ in zip(sentences[idx].split(' '), relations[idx]):
            logger.info("word_ = %s, tag_ = %s" % (word_, tag_[-1]))
        logger.info("sentence : {}".format(sentences[idx]))
        logger.info("relation : {}".format(relations[idx]))
        logger.info("-"*100)
    #raise Exception('check')
    logger.info("mode : {}, label2id : {}".format(mode,json.dumps(label2id,ensure_ascii=False)))
    for i in tqdm(range(0,len(sentences),batch_size)):
        batch_sentences=sentences[i:i+batch_size]
        batch_relations=relations[i:i+batch_size]
        inputs=tokenizer(text=batch_sentences,max_length=args.max_length,padding='max_length',truncation=True)
        batch_input_ids,batch_attention_mask,batch_token_type_ids = inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']
        for k in range(len(batch_input_ids)):
            span_mask,span_label,ner_relation=get_span_mask_label(args=args,
                                                                sentence=batch_sentences[k],
                                                                tokenizer=tokenizer,
                                                                attention_mask=batch_attention_mask[k],
                                                                relation=batch_relations[k],
                                                                label2id=label2id,
                                                                mode=mode)
            tmp = {}
            tmp['input_ids'] = batch_input_ids[k]
            tmp['token_type_ids'] = batch_token_type_ids[k]
            tmp['attention_mask'] = batch_attention_mask[k]
            tmp['span_label'] = span_label
            tmp['span_mask'] = span_mask
            tmp['input_tokens'] = batch_sentences[k]
            tmp['span_tokens'] = batch_relations[k]
            tmp['converted_span']=ner_relation
            wordpiece=[]
            for start_id,end_id,tag in ner_relation:
                wordpiece.append(' '.join(tokenizer.convert_ids_to_tokens(batch_input_ids[k][start_id:end_id+1])))
            tmp['wordpiece']=wordpiece
            data.append(tmp)
            if mode=='dp':
                for s_id,e_id,deprel in tmp['span_tokens']:
                    # print(tmp['span_tokens'])
                    # #print(span_label)
                    # print(batch_sentences[k])
                    assert span_label[s_id,e_id]==label2id[deprel]

    return data


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        one_data = {
            "input_ids": torch.tensor(item['input_ids']).long(),
            "token_type_ids": torch.tensor(item['token_type_ids']).long(),
            "attention_mask": torch.tensor(item['attention_mask']).float(),
            "span_label": torch.tensor(item['span_label']).long(),
            "span_mask": torch.tensor(item['span_mask']).long()
        }
        return one_data

def yield_data(args, file_path, tokenizer, mode, label2id, limit=None,is_training=True):
    data = data_pre(args, file_path, tokenizer, mode, label2id=label2id)
    logger.info("mode : {}, number of examples : {}".format(mode,len(data)))
    logger.info("Printing some examples...")
    for _ in range(3):
        idx=random.randint(a=0,b=len(data)-1)
        example=data[idx]
        input_ids=example['input_ids']
        token_type_ids=example['token_type_ids']
        attention_mask=example['attention_mask']
        span_label=example['span_label']
        span_mask=example['span_mask']
        input_tokens=example['input_tokens']
        span_tokens=example['span_tokens']
        logger.info("input tokens(sentence) : {}".format(input_tokens))
        logger.info("span tokens(label) : {}".format(span_tokens))
        logger.info("input_ids : {}".format(' '.join([str(i) for i in input_ids])))
        logger.info("attention_mask : {}".format(' '.join([str(i) for i in attention_mask])))

        length=sum(attention_mask)
        print_span_mask=np.array(span_mask)[:length+2,:length+2]
        print_span_label=np.array(span_label)[:length+2,:length+2]
        logger.info("Span mask : ")
        for row in print_span_mask:
            logger.info(" ".join([str(i) for i in row]))
        logger.info("Span label(wordpiece) : ")
        for row in print_span_label:
            logger.info(" ".join([str(i) for i in row])) 
        if mode=='dp':
            logger.info("deprel label position in span_label matrix : {}".format([(s_id,e_id) for s_id,e_id,deprel in span_tokens]))
        logger.info("converted_span (wordpiece): {}".format(example['converted_span']))
        logger.info("wordpiece span : ")
        for wordpiece in example['wordpiece']:
            logger.info(wordpiece)
        # logger.info("span_label : {}".format(' '.join([str(i) for i in span_label])))
        # logger.info("span_mask : {}".format(' '.join([str(i) for i in span_mask])))
        logger.info("input length: {}".format(len(input_ids)))
        logger.info('-'*100)

    logger.info("label2id : {}".format(label2id))
    if limit is not None:
        random.shuffle(data)
        data=data[:limit]
        logger.info("Limit to {}".format(limit))

    dataset=MyDataset(data=data)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=is_training, num_workers=args.num_workers)

