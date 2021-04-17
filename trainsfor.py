import json
import codecs
from tqdm import tqdm

chars = {}
min_count = 2
train_data = []

with codecs.open('./data/file.json', 'r', encoding='utf-8') as f:
    for l in tqdm(f):
        a = json.loads(l)
        text = list(a['text'])
        if len(a['labels']) == 5:
            tmp = [1, 2, 3, 4, 'end']
            for i in a['labels']:
                s = "".join(text[i[0]:i[1]])
                if i[2] == "原因中的核心名词":
                    tmp[0]=s
                if i[2] == "原因中的谓语或状态":
                    tmp[1]=s
                if i[2] == "中心词":
                    tmp[2]=s
                if i[2] == "结果中的核心名词":
                    tmp[3]=s
                if i[2] == "结果中的谓语或状态":
                    tmp[4]=s
            train_data.append(
                {
                    'text': a['text'],
                    'spo_list': [tmp]
                }
            )
            for c in a['text']:
                chars[c] = chars.get(c, 0) + 1
        else:
            continue

with codecs.open('./data/train_data_final.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

all_50_schemas = set()

with open('./data/train_data_final.json') as f:
    a = json.load(f)
    for i in a:
        for line in i['spo_list']:
            all_50_schemas.add(line[2])
id2predicate = {i:j for i,j in enumerate(all_50_schemas)}
predicate2id = {j:i for i,j in id2predicate.items()}

with codecs.open('data/all_schemas_final.json', 'w', encoding='utf-8') as f:
    json.dump([id2predicate, predicate2id], f, indent=4, ensure_ascii=False)

with codecs.open('all_chars_final.json', 'w', encoding='utf-8') as f:
    chars = {i:j for i,j in chars.items() if j >= min_count}
    id2char = {i+2:j for i,j in enumerate(chars)} # padding: 0, unk: 1
    char2id = {j:i for i,j in id2char.items()}
    json.dump([id2char, char2id], f, indent=4, ensure_ascii=False)