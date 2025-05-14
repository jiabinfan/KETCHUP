import argparse
import evaluate
import re
import os
from datasets import load_dataset
from datasets import load_metric
import json

ANS_RE = re.compile(r"#+ (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def extract_answer(completion):
    # ans_lst = re.findall(r'\d*\.?\d+', completion)
    ans_lst = re.findall(ANS_RE, completion)
    if len(ans_lst) > 0:
        try:
            ans = re.sub(',', '', ans_lst[-1])
            ans = float(ans)
        except:
            ans = INVALID_ANS
    else:
        ans = INVALID_ANS
    return ans


def parse_gold(lines):
    all_ans = []
    for line in lines:
        # ans = extract_answer(line['answer'])
        all_ans.append(line['ans'])
    return all_ans


def parse(lines):
    all_ans = []
    for line in lines:
        ans = extract_answer(line)
        all_ans.append(ans)
    return all_ans

def main():
    parser = argparse.ArgumentParser(description="Calculate ROUGE or BLEU score between two files.")
    parser.add_argument("--predictions", type=str, default='/mnt/nvme/jiabin/qwen_accLLMR/outputs/xsum_qwen1.5-0.5b_test.gen', help="File path of model predictions.")
    parser.add_argument("--references", type=str, default='data/xsum/test.tgt', help="File path of reference texts.")
    parser.add_argument("--metric", type=str, default="rouge", help="Metric to compute.")
    args = parser.parse_args()

    # Load predictions and references
    with open(args.predictions, "r", encoding="utf-8") as pred_file:
        predictions = [line.strip() for line in pred_file]

    if args.metric != "accuracy":
        with open(args.references, "r", encoding="utf-8") as ref_file:
            references = [line.strip() for line in ref_file]

        # Ensure the number of predictions and references match
        assert len(predictions) == len(references), "Predictions and references must have the same number of lines."

    # Load the specified metric
    metric = evaluate.load(args.metric)

    # Compute the score
    if args.metric == "bleu":
        # BLEU expects references to be a list of lists
        references = [[ref] for ref in references]
        results = metric.compute(predictions=predictions, references=references)
        print(f"BLEU score: {results['bleu']:.4f}")
    elif args.metric =='rouge':
        # Compute ROUGE scores
        metric = load_metric('rouge')
        scores = metric.compute(predictions=predictions, references=references, rouge_types=['rouge1', 'rouge2', 'rougeL'])
        rouges = []
        for metric, scores in scores.items():
            r = scores.high.fmeasure
            # r=scores
            rouges.append(r*100)
        print('ROUGE-1/2/L', rouges)
    else: # calculate accuracy
        generations=[]
        for gen in predictions:
            generations.append(gen)

        pred_ans=parse(generations)

        df = os.path.join('data/gsm8k', 'test.jsonl')
        gold_data = load_dataset('json', data_files=df)['train']
        lines = []
        for d in gold_data:
            lines.append(d)
        gold_ans = parse_gold(lines)

        cor = 0
        assert len(pred_ans) >= len(gold_ans)
        for i in range(len(gold_ans)):
            if pred_ans[i] != INVALID_ANS and abs(float(pred_ans[i]) - float(gold_ans[i])) < 1e-4:
                cor += 1

        accuracy=cor/len(gold_ans) * 100
        print(f'Acc: {cor}/{len(gold_ans)} = {cor/len(gold_ans) * 100:.1f}%')
            
        # refs=[item.lower().strip() for item in references]
        # accuracy = sum(g.strip() == r.strip() for g, r in zip(generations, refs)) / len(generations) if refs else 0

        # print(f"Accuracy: {accuracy:.4f}")

        return {"accuracy": accuracy}
    

if __name__ == "__main__":
    main()