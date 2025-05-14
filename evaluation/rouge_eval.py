import argparse
from nltk import wordpunct_tokenize, wordpunct_tokenize
# rouge = evaluate.load('rouge')
from datasets import load_metric
import argparse
import torch
import numpy as np
import os
import random
import evaluate

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(file, lower=False, num=None):
    strs = []
    with open(file, 'r', encoding='utf8') as of:
        if num == None:
            datas = of.readlines()
        else:
            datas = of.readlines()[:num]
        for idx, data in enumerate(datas):
            strs.append(data.strip())
    str_list = [seq for seq in strs]
    return str_list


def load_ref_data(file, length, lower=False):
    strs=[]
    with open(file,'r',encoding='utf8') as of:
        datas=of.readlines()[:length]
        for idx,data in enumerate(datas):
            strs.append(data.strip())
    str_list = [seq for seq in strs]
    return str_list

def rouge_metric(args):
    if args.num!=None:
        infer =load_data(args.gen, lower=args.lowercase, num=args.num)
    else:
        infer = load_data(args.gen, lower=args.lowercase)
    golden=load_ref_data(args.ref, len(infer), lower=args.lowercase)
    
    metric = load_metric('rouge')
    metric1 = evaluate.load('rouge')
    scores = metric.compute(predictions=infer, references=golden, rouge_types=['rouge1', 'rouge2', 'rougeL'])
    rouges = []

    for metric, scores in scores.items():
        r = scores.high.fmeasure
        # r=scores
        rouges.append(r*100)
    print('ROUGE-1/2/L', rouges)

    scores = metric1.compute(predictions=infer, references=golden, rouge_types=['rouge1', 'rouge2', 'rougeL'])
    print("metrics1: ", scores)

    return rouges[0], rouges[1], rouges[2]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', default=None, type=str)
    parser.add_argument('--gen', default=None, type=str)
    parser.add_argument('--src', default=None, type=str)
    parser.add_argument('--lowercase', action='store_true')
    parser.add_argument('--num', type=int, default=None)
    
    args = parser.parse_args()
    set_seed(0)
    rouge_metric(args)