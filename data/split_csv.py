import argparse
import csv
from transformers import AutoTokenizer

tokenizer=AutoTokenizer.from_pretrained('/mnt/nvme/jiabin/gq_accLLMR/results/sft-flan-t5-xl')

def main(args):
    prefix = '.'.join(args.file.split('.')[:-1])

    with open(args.file) as fp, \
        open(prefix + '.src', 'w') as ofp1, \
        open(prefix + '.tgt', 'w') as ofp2:
        reader = csv.reader(fp)
        para = 0
        mono = 0
        for i, row in enumerate(reader):
            if i == 0:
                k = row
            else:
                d = {k[j]: row[j] for j in range(len(row))}
                ofp1.write(repr(d['context']).strip("'").strip('"').strip() + '\n')
                ofp2.write(repr(d['response']).strip("'").strip('"').strip() + '\n')
                para += 1
        print(para, mono)

def f1(args):
    prefix = args.file
    ds=args.dataset+'/'
    data = []
    tgt = []
    src = []
    print(ds+prefix+".tgt")
    max_length=0

    with open(ds+prefix+".tgt", "r") as fi1:
        for line in fi1:
            tgt.append(line.strip("\n"))
            
    with open(ds+prefix+".src", "r") as fi2:
        if 'gsm8k' not in ds:
            for line in fi2:
                src.append(line.strip("\n"))
                # input_tokens = tokenizer.encode(line, add_special_tokens=False)
                # # Update the maximum length
                # max_length = max(max_length, len(input_tokens))
        else:
            for line in fi2:
                line=line.strip("\n")
                line=f"Question: {line}\nAnswer: "
                input_tokens = tokenizer.encode(line, add_special_tokens=False)
                # Update the maximum length
                max_length = max(max_length, len(input_tokens))

                src.append(line)
        # print(f"Max length of input sequences: {max_length}")
            
    for i in range(len(tgt)):
        if 'gsm8k' in ds:
            if len(src[i]) < 1 or len(tgt[i]) < 1:
                print(i)
                continue
        else:
            if len(src[i]) < 2 or len(tgt[i]) < 2:
                print(i)
                continue
        data.append({"document":src[i], "summary":tgt[i]})
        # if i > 100:
        #     break
    with open(ds+prefix+'.csv', 'w', newline='') as csvfile:
        fieldnames = ['document', 'summary']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
        print("csv file created")
        print(len(data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='valid')
    parser.add_argument("--dataset", type=str, default='xsum')

    args = parser.parse_args()
    f1(args)