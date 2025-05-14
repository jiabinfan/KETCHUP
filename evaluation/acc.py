import argparse
from nltk import wordpunct_tokenize, wordpunct_tokenize
# rouge = evaluate.load('rouge')
from datasets import load_metric
import argparse
import torch
import numpy as np
import os
import random


def cal_acc(args):
    gens=open(args.gen, encoding='utf8').readlines()
    generations=[]
    for gen in gens:
        gen=gen.strip().lower()
        generation=gen.split('answer is:')[-1].split('.')[0].strip()
        generations.append(generation)

    refs=[ref.strip().lower() for ref in open(args.ref, encoding='utf8').readlines()]

    accuracy = sum(g.strip() == r.strip() for g, r in zip(generations, refs)) / len(generations) if refs else 0

    print(f'Accuracy: {accuracy * 100:.4f}%')

    return accuracy



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', default='data/gsm8k/test.tgt', type=str)
    parser.add_argument('--gen', default='flan-t5-xl.gen', type=str)
    parser.add_argument('--src', default=None, type=str)
    parser.add_argument('--lowercase', action='store_true')
    
    args = parser.parse_args()
    cal_acc(args)
