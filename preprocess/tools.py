import os
import sys
sys.path.append('./')
from transformers import AutoTokenizer
import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import unicodedata, re

def get_tokenizer(bert_model_path):
    """[unused1] token"""
    tokenizer=AutoTokenizer.from_pretrained(bert_model_path)
    return tokenizer

def generate_label2id(file_path):
    with open(file_path) as f:
        lines=f.readlines()
    label2id={}
    for line in lines:
        line_split=line.strip().split()
        if len(line_split)>1:
            label2id[line_split[-1]]=len(label2id)
    return label2id

def process_nerlabel(label2id):
    #label2id,id2label,num_labels = tools.load_schema_ner()
    #Since different ner dataset has different entity categories, it is inappropriate to pre-assign entity labels
    new_={}
    new_={'O':0}
    for label in label2id:
        if label!='O':
            label='-'.join(label.split('-')[1:])
            if label not in new_:
                new_[label]=len(new_)
    return new_

class token_rematch:
    def __init__(self):
        self._do_lower_case = True


    @staticmethod
    def stem(token):
            """strip ##
            """
            if token[:2] == '##':
                return token[2:]
            else:
                return token
    @staticmethod
    def _is_control(ch):
            """control token process
            """
            return unicodedata.category(ch) in ('Cc', 'Cf')
    @staticmethod
    def _is_special(ch):
            """other symbolic token
            """
            return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    def rematch(self, text, tokens):
        """token mapping
        """
        if self._do_lower_case:
            text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping


def search(pattern, sequence):
    """find sub pattern
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


def load_schema_dp(mode = 'Ontonotes'):
    # load schema
    assert mode in ['Ontonotes', 'PTB'], mode
    
    if mode == 'PTB':
        label2id={
            'prep': 1,
            'det': 2,
            'nn': 3,
            'num': 4,
            'pobj': 5,
            'punct': 6,
            'poss': 7,
            'possessive': 8,
            'amod': 9,
            'nsubj': 10,
            'dep': 11,
            'dobj': 12,
            'cc': 13,
            'conj': 14,
            'nsubjpass': 15,
            'partmod': 16,
            'auxpass': 17,
            'advmod': 18,
            'root': 19,
            'ccomp': 20,
            'aux': 21,
            'cop': 22,
            'xcomp': 23,
            'quantmod': 24,
            'tmod': 25,
            'appos': 26,
            'npadvmod': 27,
            'neg': 28,
            'infmod': 29,
            'rcmod': 30,
            'pcomp': 31,
            'mark': 32,
            'advcl': 33,
            'predet': 34,
            'csubj': 35,
            'mwe': 36,
            'parataxis': 37,
            'number': 38,
            'acomp': 39,
            'prt': 40,
            'iobj': 41,
            'expl': 42,
            'preconj': 43,
            'discourse': 44,
            'csubjpass': 45}
        
        id2label={
            1: 'prep',
            2: 'det',
            3: 'nn',
            4: 'num',
            5: 'pobj',
            6: 'punct',
            7: 'poss',
            8: 'possessive',
            9: 'amod',
            10: 'nsubj',
            11: 'dep',
            12: 'dobj',
            13: 'cc',
            14: 'conj',
            15: 'nsubjpass',
            16: 'partmod',
            17: 'auxpass',
            18: 'advmod',
            19: 'root',
            20: 'ccomp',
            21: 'aux',
            22: 'cop',
            23: 'xcomp',
            24: 'quantmod',
            25: 'tmod',
            26: 'appos',
            27: 'npadvmod',
            28: 'neg',
            29: 'infmod',
            30: 'rcmod',
            31: 'pcomp',
            32: 'mark',
            33: 'advcl',
            34: 'predet',
            35: 'csubj',
            36: 'mwe',
            37: 'parataxis',
            38: 'number',
            39: 'acomp',
            40: 'prt',
            41: 'iobj',
            42: 'expl',
            43: 'preconj',
            44: 'discourse',
            45: 'csubjpass'}
        
        num_labels=45

    else:
        label2id={
            'det': 1,
            'root': 2,
            'prep': 3,
            'pobj': 4,
            'punct': 5,
            'nsubj': 6,
            'advmod': 7,
            'dobj': 8,
            'aux': 9,
            'vmod': 10,
            'amod': 11,
            'nn': 12,
            'appos': 13,
            'acomp': 14,
            'nsubjpass': 15,
            'auxpass': 16,
            'conj': 17,
            'dep': 18,
            'cc': 19,
            'num': 20,
            'cop': 21,
            'poss': 22,
            'possessive': 23,
            'xcomp': 24,
            'ccomp': 25,
            'prt': 26,
            'rcmod': 27,
            'discourse': 28,
            'mwe': 29,
            'pcomp': 30,
            'mark': 31,
            'npadvmod': 32,
            'advcl': 33,
            'predet': 34,
            'parataxis': 35,
            'neg': 36,
            'tmod': 37,
            'quantmod': 38,
            'number': 39,
            'expl': 40,
            'iobj': 41,
            'csubj': 42,
            'preconj': 43,
            'csubjpass': 44,
            'erased': 45
            }
        id2label={id_:tag for tag, id_ in label2id.items()}
        num_labels=45

    return label2id,id2label,num_labels


# def load_schema_ner():
#     label2id={
#     'ORG_whole': 1,
#     'ORG_sub': 2,
#     'WORK_OF_ART_whole': 3,
#     'WORK_OF_ART_sub': 4,
#     'LOC_whole': 5,
#     'LOC_sub': 6,
#     'CARDINAL_whole': 7,
#     'CARDINAL_sub': 8,
#     'EVENT_whole': 9,
#     'EVENT_sub': 10,
#     'NORP_whole': 11,
#     'NORP_sub': 12,
#     'GPE_whole': 13,
#     'GPE_sub': 14,
#     'DATE_whole': 15,
#     'DATE_sub': 16,
#     'PERSON_whole': 17,
#     'PERSON_sub': 18,
#     'FAC_whole': 19,
#     'FAC_sub': 20,
#     'QUANTITY_whole': 21,
#     'QUANTITY_sub': 22,
#     'ORDINAL_whole': 23,
#     'ORDINAL_sub': 24,
#     'TIME_whole': 25,
#     'TIME_sub': 26,
#     'PRODUCT_whole': 27,
#     'PRODUCT_sub': 28,
#     'PERCENT_whole': 29,
#     'PERCENT_sub': 30,
#     'MONEY_whole': 31,
#     'MONEY_sub': 32,
#     'LAW_whole': 33,
#     'LAW_sub': 34,
#     'LANGUAGE_whole': 35,
#     'LANGUAGE_sub': 36}
    
#     id2label={
#     1: 'ORG_whole',
#     2: 'ORG_sub',
#     3: 'WORK_OF_ART_whole',
#     4: 'WORK_OF_ART_sub',
#     5: 'LOC_whole',
#     6: 'LOC_sub',
#     7: 'CARDINAL_whole',
#     8: 'CARDINAL_sub',
#     9: 'EVENT_whole',
#     10: 'EVENT_sub',
#     11: 'NORP_whole',
#     12: 'NORP_sub',
#     13: 'GPE_whole',
#     14: 'GPE_sub',
#     15: 'DATE_whole',
#     16: 'DATE_sub',
#     17: 'PERSON_whole',
#     18: 'PERSON_sub',
#     19: 'FAC_whole',
#     20: 'FAC_sub',
#     21: 'QUANTITY_whole',
#     22: 'QUANTITY_sub',
#     23: 'ORDINAL_whole',
#     24: 'ORDINAL_sub',
#     25: 'TIME_whole',
#     26: 'TIME_sub',
#     27: 'PRODUCT_whole',
#     28: 'PRODUCT_sub',
#     29: 'PERCENT_whole',
#     30: 'PERCENT_sub',
#     31: 'MONEY_whole',
#     32: 'MONEY_sub',
#     33: 'LAW_whole',
#     34: 'LAW_sub',
#     35: 'LANGUAGE_whole',
#     36: 'LANGUAGE_sub'}
    
#     num_labels=36
    
#     return label2id,id2label,num_labels

def load_schema_ner():
    
    id2label={1: 'ORG_whole',
            2: 'WORK_OF_ART_whole',
            3: 'LOC_whole',
            4: 'CARDINAL_whole',
            5: 'EVENT_whole',
            6: 'NORP_whole',
            7: 'GPE_whole',
            8: 'DATE_whole',
            9: 'PERSON_whole',
            10: 'FAC_whole',
            11: 'QUANTITY_whole',
            12: 'ORDINAL_whole',
            13: 'TIME_whole',
            14: 'PRODUCT_whole',
            15: 'PERCENT_whole',
            16: 'MONEY_whole',
            17: 'LAW_whole',
            18: 'LANGUAGE_whole'
            }
    label2id={k:v for v,k in id2label.items()}
    
    num_labels=len(label2id)
    
    return label2id,id2label,num_labels

def batch_to_device(tensor_dicts,device):
    for key in tensor_dicts.keys():
        tensor_dicts[key].to(device)

def solve_wordpiece(last_hidden_state,sen):
    '''
    sen is a sentence in type of string
    last_hidden_state not contains CLS and SEP
    tokenizer.tokenize is not equal to tokenizer.convert_tokens_to_ids
    like sentence : 'Total shares to be offered 0.0 million'
    input_ids :[8653, 6117, 1106, 1129, 2356, 121, 119, 121, 1550]
    word_ids : [8653, 6117, 1106, 1129, 2356, 100, 1550]           100 means UNK
    '''
    new_state=[]
    if type(sen)==str:
        sentence_list=sen.split(' ')
    else:
        sentence_list=sen
    j=0
    for i in range(len(sentence_list)):
        token=sentence_list[i]
        tokens=tokenizer.tokenize(token)
        piece_length=len(tokens)
        new_state.append(torch.mean(last_hidden_state[j:j+piece_length],dim=0,keepdims=True))
        j+=piece_length
    new_state=torch.vstack(new_state)
    assert new_state.size(0)==len(sentence_list)
    return new_state

