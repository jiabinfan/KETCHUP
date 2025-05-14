from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk import wordpunct_tokenize, wordpunct_tokenize
from nltk.collocations import BigramCollocationFinder
from nltk.probability import FreqDist
import argparse
import torch
import numpy as np
import os
import random
from evaluate import load

# from sacrebleu import corpus_bleu, corpus_chrf, corpus_ter
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

BLEU_WEIGHTS_MEAN = [
    [1.0],
    [0.5, 0.5],
    [1/3, 1/3, 1/3],
    [0.25, 0.25, 0.25, 0.25],
]

SMOOTHING_FUNCTION = SmoothingFunction().method0

def calculate_ngram_diversity(corpus):
    """
    Calculates unigram and bigram diversity
    Args:
        corpus: tokenized list of sentences sampled
    Returns:
        uni_diversity: distinct-1 score
        bi_diversity: distinct-2 score
    """
    bigram_finder = BigramCollocationFinder.from_words(corpus)
    try:
        bi_diversity = len(bigram_finder.ngram_fd) / bigram_finder.N
    except ZeroDivisionError:
        print('Division by zero in dist-2 calcluation')
        bi_diversity = 0
    dist = FreqDist(corpus)
    try:
        uni_diversity = len(dist) / len(corpus)
    except ZeroDivisionError:
        print('Division by zero in dist-1 calcluation')
        uni_diversity = 0
    return uni_diversity, bi_diversity


def i_corpus_bleu(
    list_of_references,
    hypotheses,
    inputs,
    alpha=0.9,
    weights=(0.25, 0.25, 0.25, 0.25),
    smoothing_function=None,
    auto_reweigh=False,
):
    list_of_inputs = [[i] for i in inputs]
    bleu = corpus_bleu(
        list_of_references,
        hypotheses,
        weights=weights,
        smoothing_function=smoothing_function,
        auto_reweigh=auto_reweigh,
    )
    penalty = corpus_bleu(
        list_of_inputs,
        hypotheses,
        weights=weights,
        smoothing_function=smoothing_function,
        auto_reweigh=auto_reweigh,
    )
    return alpha * bleu - (1 - alpha) * penalty


def load_data(file, lower=False):
    strs = []
    with open(file, 'r', encoding='utf8') as of:
        datas = of.readlines()
        for idx, data in enumerate(datas):
            strs.append(data.strip())
    str_list = [wordpunct_tokenize(seq.lower() if lower else seq) for seq in strs]
    print(str_list)[:5]
    return str_list,strs


def load_ref_data(file, lower=False):
    strs=[]
    with open(file,'r',encoding='utf8') as of:
        datas=of.readlines()
        for idx,data in enumerate(datas):
            strs.append(data.strip())
    str_list = [[wordpunct_tokenize(seq.lower() if lower else seq)] for seq in strs]
    print(str_list)[0][:5]
    return str_list,strs


def metric(args):
    infer, infer_list =load_data(args.gen, lower=args.lowercase)
    # ori= load_data(args.src, lower=args.lowercase)
    golden, gold_list=load_ref_data(args.ref, lower=args.lowercase)
    
    # eval bleu
    corp_model_bleu1 = corpus_bleu(golden, infer, weights=BLEU_WEIGHTS_MEAN[0]) * 100
    corp_model_bleu2 = corpus_bleu(golden, infer, weights=BLEU_WEIGHTS_MEAN[1]) * 100
    corp_model_bleu3 = corpus_bleu(golden, infer, weights=BLEU_WEIGHTS_MEAN[2]) * 100
    corp_model_bleu4 = corpus_bleu(golden, infer, weights=BLEU_WEIGHTS_MEAN[3]) * 100
    nltk_bleu=[corp_model_bleu1, corp_model_bleu2, corp_model_bleu3, corp_model_bleu4]
    print('BLEU', nltk_bleu)
    
    # eval dist
    tokens=[token for sentence in infer for token in sentence]
    dist_1, dist_2 = calculate_ngram_diversity(tokens)
    # distinct=[dist_1 * 100, dist_2 * 100]
    # print('distinct', distinct)

    # print('length-gen', sum(1 for sentence in infer for _ in sentence))
    # print('length-ref', sum(1 for ref in golden for sent in ref for _ in sent))

    # bert_score = load('bertscore')
    # bertscore = bert_score.compute(predictions=infer_list, references=gold_list, model_type="roberta-base", lang="en", rescale_with_baseline=True)
    # bertscore_f1 = sum(bertscore['f1']) / len(bertscore['f1'])
    # print('bertscore', bertscore_f1)

    return corp_model_bleu2, corp_model_bleu4, dist_1, dist_2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', default=None, type=str)
    parser.add_argument('--gen', default=None, type=str)
    parser.add_argument('--src', default=None, type=str)
    parser.add_argument('--lowercase', action='store_true')
    
    args = parser.parse_args()
    set_seed(0)
    metric(args)
