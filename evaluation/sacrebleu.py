from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk import wordpunct_tokenize, wordpunct_tokenize
from nltk.collocations import BigramCollocationFinder
from nltk.probability import FreqDist
import argparse

from sacrebleu import corpus_bleu


def metric(args):

    def calculate_test_bleu(gen_file, ref_file, **kwargs) -> dict:
        """Uses sacrebleu's corpus_bleu implementation."""
        # Read generated translations
        with open(gen_file, 'r', encoding='utf-8') as gen_f:
            output_lns = [line.strip() for line in gen_f]

        # Read reference translations
        with open(ref_file, 'r', encoding='utf-8') as ref_f:
            reference_lns = [line.strip() for line in ref_f]

        assert len(output_lns) == len(reference_lns), \
            f"The number of hypotheses ({len(output_lns)}) and references ({len(reference_lns)}) should be the same"
        
        # Calculate BLEU score
        bleu_score = corpus_bleu(output_lns, [reference_lns], **kwargs)
        print("BLEU score: ", bleu_score.score)
        return {"bleu_score": bleu_score.score}
    
    calculate_test_bleu(args.gen, args.ref)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', default=None, type=str)
    parser.add_argument('--gen', default=None, type=str)
    parser.add_argument('--src', default=None, type=str)
    parser.add_argument('--lowercase', action='store_true')
    
    args = parser.parse_args()
    metric(args)
