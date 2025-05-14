import json
import csv
with open('train.jsonl', 'r', encoding='utf-8') as f1, open('train.csv', 'w', encoding='utf-8') as f2:
    f2.write("question,answer\n")
    new_datas=[]
    for line in f1:
        data = json.loads(line)
        new_data={}
        new_data['document']=data['question'].strip()
        new_data['summary']=data['answer'].strip()
        new_datas.append(new_data)
    fieldnames = ['document', 'summary']
    writer = csv.DictWriter(f2, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(new_datas)
    print("csv file created")
    print(len(new_datas))
    
        