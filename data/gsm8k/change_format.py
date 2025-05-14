import re
import json
import pandas as pd

def process_answer(solution):
    solution = re.sub(r'<<\d+=\d+>>', '', solution)
    solution = re.sub(r'(\D):(\d)', r'\1: \2', solution)
    solution = re.sub(r'<<', '[ ', solution)
    solution = re.sub(r'>>', '] ', solution)
    solution = re.sub(r'\$([^\s])', r'$ \1', solution)
    solution = re.sub(r'([^\s])\+([^\s])', r'\1 + \2', solution)
    solution = re.sub(r'([^\s])-([^\s])', r'\1 - \2', solution)
    solution = re.sub(r'([^\s])\*([^\s])', r'\1 * \2', solution)
    solution = re.sub(r'([^\s])/([^\s])', r'\1 / \2', solution)
    solution = re.sub(r'([^\s])=([^\s])', r'\1 = \2', solution)
    solution_split_by_line = solution.split('\n')
    sol = []
    for l in solution_split_by_line:
        if l.startswith('##'):
            sol.append(l)
        elif not l.endswith('.'):
            sol.append(l+'.')
        else:
            sol.append(l)
    solution = ' '.join(sol)
    return solution

def process_file(prefix):
    data=[]
    with open(f'data/gsm8k/{prefix}.jsonl') as f:
        for line in f:
            data.append(json.loads(line))
    
    with open(f'data/gsm8k/{prefix}.src','w') as f:
        for i, line in enumerate(data):
            f.write(data[i]['question']+'\n')
    with open(f'data/gsm8k/{prefix}.tgt','w') as f:
        for i, line in enumerate(data):
            f.write(process_answer(data[i]['answer'])+'\n')
    df = pd.DataFrame(data)
    df['document']=df['question']
    df['summary'] = df['answer'].apply(process_answer)
    df = df.drop(columns=['question', 'answer'])  # 使用 drop 方法删除列
    df = df[['document', 'summary', 'ans']]  # 重新排列列的顺序
    df.to_csv(f'data/gsm8k/{prefix}.csv', index=False)


process_file('test')
# process_file('train')
# process_file('valid')