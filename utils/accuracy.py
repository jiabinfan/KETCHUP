import evaluate
import datasets
import re 
import os
from datasets import load_dataset

_DESCRIPTION = """\
"""
_CITATION = """\
"""
_KWARGS_DESCRIPTION = """\
"""

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

class Acc(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            # inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Sequence(datasets.Value("string", id="sequence")),
                    }
                ),
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Value("string", id="sequence"),
                    }
                ),
            ],
            codebase_urls=["https://github.com/google-research/google-research/tree/master/rouge"],
            reference_urls=[
                "https://en.wikipedia.org/wiki/ROUGE_(metric)",
                "https://github.com/google-research/google-research/tree/master/rouge",
            ],
        )

    def _compute(self, predictions, references, tokenizer=None):
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
        print(f'len(pred_ans): {len(pred_ans)}', f'len(gold_ans): {len(gold_ans)}')
        assert len(pred_ans) >= len(gold_ans)
        for i in range(len(gold_ans)):
            if pred_ans[i] != INVALID_ANS and abs(float(pred_ans[i]) - float(gold_ans[i])) < 1e-4:
                cor += 1

        accuracy=cor/len(gold_ans) * 100
        print(f'Acc: {cor}/{len(gold_ans)} = {cor/len(gold_ans) * 100:.2f}%')
            
        # refs=[item.lower().strip() for item in references]
        # accuracy = sum(g.strip() == r.strip() for g, r in zip(generations, refs)) / len(generations) if refs else 0

        # print(f"Accuracy: {accuracy:.4f}")

        return {"accuracy": accuracy}

