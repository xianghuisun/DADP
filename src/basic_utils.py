import logging, argparse
import torch
import random
import numpy as np
    
def init_logger(logger_name, log_file_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG) 
    file_handler = logging.FileHandler(log_file_name,mode='w')
    file_handler.setLevel(logging.INFO) 
    file_handler.setFormatter(
            logging.Formatter(
                    fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
            )
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
            logging.Formatter(
                    fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
            )
    logger.addHandler(console_handler)
    return logger


def init_args(mode='dp'):
    parser = argparse.ArgumentParser()
    if mode == 'dp':
        parser.add_argument("--dp_train_path", type=str, required=True, help="train file")
        parser.add_argument("--dp_test_path", type=str, required=True, help="test file")
        parser.add_argument("--dp_mode", type=str, required=True, help="Ontonotes or PTB")
    else:
        parser.add_argument("--ner_train_path", type=str, required=True, help="train file")
        parser.add_argument("--ner_test_path", type=str, required=True, help="test file")
        parser.add_argument("--dp_model_checkpoints", type=str, required=True, help="pretrained dp model")
        parser.add_argument("--dataset_name", type=str, required=True, help="dataset_name")
        
    parser.add_argument("--checkpoints", type=str, required=True, help="output_dir")
    parser.add_argument("--batch_size", type=int, default=32,help="batch_size")
    parser.add_argument("--lstm_hidden_size", type=int, default=512,help="lstm_hidden_size")
    parser.add_argument("--to_biaffine_size", type=int, default=128,help="to_biaffine_size")
    parser.add_argument("--max_length", type=int, default=256,help="max_length")
    parser.add_argument("--epoch", type=int, default=100,help="epoch")
    parser.add_argument("--learning_rate", type=float, default=5e-5,help="learning_rate")
    parser.add_argument("--pretrained_model_path", type=str, default="/data/aisearch/nlp/data/xhsun/huggingfaceModels/english/bert_large_wwm",help="pretrained_model_path")
    parser.add_argument("--clip_norm", type=float, default=1,help="clip_norm")
    parser.add_argument("--warmup_proportion", type=float, default=0.08,help="warmup proportion")
    parser.add_argument("--num_workers", type=int, default=8,help='num_workers')
    args = parser.parse_args()
    return args

def init_seed(seed = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # System based
    random.seed(seed)
    np.random.seed(seed)
