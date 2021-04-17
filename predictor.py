# -*- coding:utf-8 -*-
import extract_maybe
import json
import codecs
from tqdm import tqdm

dev_data = []

with codecs.open('./data/office.json', 'r', encoding='utf-8') as f:
    for l in tqdm(f):
        a = json.loads(l)
        for doc in a['document']:
            for i in a['qas']:
                tmp = [1, 2, 3, 4, 'end']
                for j in i:
                    for k in j['answers']:
                        if j['question'] == "原因中的核心名词":
                            tmp[0] = k['text']
                        if j['question'] == "原因中的谓语或状态":
                            tmp[1] = k['text']
                        if j['question'] == "中心词":
                            tmp[2] = k['text']
                        if j['question'] == "结果中的核心名词":
                            tmp[3] = k['text']
                        if j['question'] == "结果中的谓语或状态":
                            tmp[4] = k['text']
            dev_data.append(
                {
                    'text': doc['text'],
                    'spo_list': [tmp]
                }
            )

# with codecs.open('./data/dev_data.json', 'w', encoding='utf-8') as f:
#     json.dump(dev_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # print(extract_maybe.extract_items("08年4月，郑煤集团拟以非公开发行的方式进行煤炭业务整体上市，解决与郑州煤电同业竞争问题，但之后由于股市的大幅下跌导致股价跌破发行价而被迫取消整体上市。"))
    orders = ['subject', 'ssubject', 'predicate', 'object', 'oobject']
    A, B, C = 1e-10, 1e-10, 1e-10
    F = open('dev_pred.json', 'w')
    for d in dev_data:
        R = set(extract_maybe.extract_items(d['text'][:512]))
        T = set()
        for iterm in d['spo_list']:
            T.add(tuple(iterm))
        A += len(R & T)
        B += len(R)
        C += len(T)
        s = json.dumps({
            'text': d['text'],
            'spo_list': [
                dict(zip(orders, spo)) for spo in T
            ],
            'spo_list_pred': [
                dict(zip(orders, spo)) for spo in R
            ],
            'new': [
                dict(zip(orders, spo)) for spo in R - T
            ],
            'lack': [
                dict(zip(orders, spo)) for spo in T - R
            ]
        }, ensure_ascii=False, indent=4)
        F.write(s + '\n')
    F.close()
    print('f1: %.4f, precision: %.4f, recall: %.4f\n' % (2 * A / (B + C), A / B, A / C))