from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import tqdm

json_path = Path('rainbow_bank/jsonl') # /nfs/home/duan/texcompile/
dataset = []
jsonls = []
for jsonl in json_path.glob('*.jsonl'):
    jsonls.append(jsonl)

for jsonl in tqdm.tqdm(jsonls):
    with open(jsonl) as file:
        for line in file:
            d = json.loads(line)
            if d['label_segment_order']:
                dataset.append(json.dumps(d))

X_train, X_test = train_test_split(dataset, test_size=0.1, random_state=2024)

with open('rainbow_bank/train.jsonl', 'w') as file:
    file.write('\n'.join(X_train))
with open('rainbow_bank/test.jsonl', 'w') as file:
    file.write('\n'.join(X_test))
print(len(X_train), len(X_test))