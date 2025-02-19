from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import tqdm

data_path = Path('/raid/duan/cd58hofa/rainbow_bank_edit/')
jsonl_path = data_path/'jsonl' # /nfs/home/duan/texcompile/
json_path = data_path/'json'
json_path.mkdir(exist_ok=True)

dataset = []
jsonls = []
for jsonl in jsonl_path.glob('*.jsonl'):
    jsonls.append(jsonl)

for jsonl in tqdm.tqdm(jsonls):
    with open(jsonl) as file:
        for line in file:
            d = json.loads(line)
            if d['label_segment_order']:
                uid = d['uid']
                with open(json_path/(uid+'.json'), 'w') as sample:
                    json.dump(d, sample)
                dataset.append(str(json_path/(uid+'.json')))

X_train, X_test = train_test_split(dataset, test_size=0.1, random_state=2024)
with open(data_path/'train.txt', 'w') as file:
    file.write('\n'.join(X_train))
with open(data_path/'test.txt', 'w') as file:
    file.write('\n'.join(X_test))
print(len(X_train), len(X_test))