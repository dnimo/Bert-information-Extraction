#! -*- coding:utf-8 -*-

import json
import numpy as np
from random import choice
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import os
import codecs

mode = 0
maxlen = 128
char_size = 128
learning_rate = 5e-5
min_learning_rate = 1e-5

config_path = './chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_L-12_H-768_A-12/vocab.txt'

token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


# 重写tokenizer
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)

train_data = json.load(open('./data/train_data_final.json'))
id2predicate, predicate2id = json.load(open('./data/all_schemas_final.json'))
id2predicate = {int(i): j for i, j in id2predicate.items()}
num_classes = len(id2predicate)

total_data = []
total_data.extend(train_data)

if not os.path.exists('./data/random_order_train.json'):
    random_order = list(range(len(total_data)))
    np.random.shuffle(random_order)
    json.dump(
        random_order,
        open('./data/random_order_train.json', 'w'),
        indent=4
    )
else:
    random_order = json.load(open('./data/random_order_train.json'))

train_data = [total_data[j] for i, j in enumerate(random_order) if i % 8 != mode]
dev_data = [total_data[j] for i, j in enumerate(random_order) if i % 8 == mode]

predicates = {}

for d in train_data:
    for sp in d['spo_list']:
        if sp[2] not in predicates:
            predicates[sp[2]] = []
        predicates[sp[2]].append(sp)


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array(
        [
            np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
        ]
    )


def list_find(list1, list2):
    n_list2 = len(list2)
    for i in list(range(len(list1))):
        if list1[i: i + n_list2] == list2:
            return i
    return -1


class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            T1, T2, S1, S2, S3, S4, K1, K2, K3, K4, O1, O2, O3, O4, OK1, OK2, OK3, OK4 = [], [], [], [], [], [], [], \
                                                                                         [], [], [], [], [], [], [], [], [], [], []
            for i in idxs:
                d = self.data[i]
                text = d['text'][:maxlen]
                tokens = tokenizer.tokenize(text)
                items = {}
                for sp in d['spo_list']:
                    sp = (tokenizer.tokenize(sp[0])[1:-1], tokenizer.tokenize(sp[1])[1:-1], sp[2],
                          tokenizer.tokenize(sp[3])[1:-1], tokenizer.tokenize(sp[4])[1:-1])
                    subjectid = list_find(tokens, sp[0])
                    ssubjectid = list_find(tokens, sp[1])
                    objectid = list_find(tokens, sp[3])
                    oobjectid = list_find(tokens, sp[4])
                    if subjectid != -1 and ssubjectid != -1 and objectid != -1 and oobjectid != -1:
                        key = (subjectid, subjectid + len(sp[0]), ssubjectid, ssubjectid + len(sp[1]))
                        if key not in items:
                            items[key] = []
                        items[key].append((objectid,
                                           objectid + len(sp[3]),
                                           oobjectid,
                                           oobjectid + len(sp[4]),
                                           predicate2id[sp[2]]))
                if items:
                    t1, t2 = tokenizer.encode(first=text)
                    T1.append(t1)
                    T2.append(t2)
                    s1, s2, s3, s4 = np.zeros(len(tokens)), np.zeros(len(tokens)), np.zeros(len(tokens)), np.zeros(
                        len(tokens))
                    # 追加sub_Task
                    for j in items:
                        s1[j[0]] = 1
                        s2[j[1] - 1] = 1
                        s3[j[2]] = 1
                        s4[j[3] - 1] = 1
                    k1, k2, k3, k4 = np.array(list(items.keys())).T
                    k1 = choice(k1)
                    k2 = choice(k2[k2 >= k1])
                    k3 = choice(k3)
                    k4 = choice(k4[k4 >= k3])
                    o1, o2, o3, o4 = np.zeros((len(tokens), num_classes)), np.zeros((len(tokens), num_classes)), \
                                     np.zeros((len(tokens), num_classes)), np.zeros((len(tokens), num_classes))
                    # 追加object_Task
                    for j in items.get((k1, k2, k3, k4), []):
                        o1[j[0]][j[4]] = 1
                        o2[j[1] - 1][j[4]] = 1
                        o3[j[2]][j[4]] = 1
                        o4[j[3] - 1][j[4]] = 1
                    ok1, ok2, ok3, ok4, nk = np.array(items.get((k1, k2, k3, k4), [])).T
                    ok1 = choice(ok1)
                    ok2 = choice(ok2[ok2 >= ok1])
                    S1.append(s1)
                    S2.append(s2)
                    S3.append(s3)
                    S4.append(s4)
                    K1.append([k1])
                    K2.append([k2 - 1])
                    K3.append([k3])
                    K4.append([k4 - 1])
                    O1.append(o1)
                    O2.append(o2)
                    O3.append(o3)
                    O4.append(o4)
                    OK1.append([ok1])
                    OK2.append([ok2 - 1])
                    if len(T1) == self.batch_size or i == idxs[-1]:
                        T1 = seq_padding(T1)
                        T2 = seq_padding(T2)
                        S1 = seq_padding(S1)
                        S2 = seq_padding(S2)
                        S3 = seq_padding(S3)
                        S4 = seq_padding(S4)
                        O1 = seq_padding(O1, np.zeros(num_classes))
                        O2 = seq_padding(O2, np.zeros(num_classes))
                        O3 = seq_padding(O3, np.zeros(num_classes))
                        O4 = seq_padding(O4, np.zeros(num_classes))
                        K1, K2, K3, K4 = np.array(K1), np.array(K2), np.array(K3), np.array(K4)
                        OK1, OK2 = np.array(OK1), np.array(OK2)
                        yield [T1, T2, S1, S2, S3, S4, K1, K2, K3, K4, O1, O2, O3, O4, OK1, OK2], None
                        T1, T2, S1, S2, S3, S4, K1, K2, K3, K4, O1, O2, O3, O4, OK1, OK2 = [], [], [], [], [], [], [], \
                                                                                           [], [], [], [], [], [], [], [], []


# Bert预训练模型开始

from keras.layers import *
from keras.models import Model
import keras.backend as K
import tensorflow as tf
from keras.callbacks import Callback
from keras.optimizers import Adam


def seq_gather(x):
    # seq是[none,seq_len,s_size]的格式，idxs是[None,1]的格式
    # 在seq的第i个序列中选出第i个向量，最终输出[None,s_size]的向量
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')
    batch_idxs = K.arange(0, K.shape(seq)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, idxs], 1)
    return tf.gather_nd(seq, idxs)


bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

for l in bert_model.layers:
    l.trainable = True

t1_in = Input(shape=(None,))
t2_in = Input(shape=(None,))
s1_in = Input(shape=(None,))
s2_in = Input(shape=(None,))
s3_in = Input(shape=(None,))
s4_in = Input(shape=(None,))
k1_in = Input(shape=(1,))
k2_in = Input(shape=(1,))
k3_in = Input(shape=(1,))
k4_in = Input(shape=(1,))
o1_in = Input(shape=(None, num_classes))
o2_in = Input(shape=(None, num_classes))
o3_in = Input(shape=(None, num_classes))
o4_in = Input(shape=(None, num_classes))
ok1_in = Input(shape=(1,))
ok2_in = Input(shape=(1,))

t1, t2, s1, s2, s3, s4, k1, k2, k3, k4, o1, o2, o3, o4 = t1_in, t2_in, s1_in, s2_in, s3_in, s4_in, k1_in, k2_in, k3_in, k4_in, o1_in, o2_in, o3_in, o4_in
ok1, ok2 = ok1_in, ok2_in

mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(t1)


def position_id(x):
    if isinstance(x, list) and len(x) == 2:
        x, r = x
    else:
        r = 0
    pid = K.arange(K.shape(x)[1])
    pid = K.expand_dims(pid, 0)
    pid = K.tile(pid, [K.shape(x)[0], 1])
    return K.abs(pid - K.cast(r, 'int32'))


pid = Lambda(position_id)([t1])
position_embedding = Embedding(maxlen, char_size, embeddings_initializer='zeros')
pv = position_embedding(pid)

t = bert_model([t1, t2])
t = Add()([t, pv])
ps1 = Dense(char_size, activation='relu')(t)
ps1 = Dense(1, activation='sigmoid')(ps1)
ps2 = Dense(char_size, activation='relu')(t)
ps2 = Dense(1, activation='sigmoid')(ps2)

subject_model = Model([t1_in, t2_in], [ps1, ps2])

k1v = position_embedding(Lambda(position_id)([t, k1]))
k2v = position_embedding(Lambda(position_id)([t, k2]))
kv = Average()([k1v, k2v])
t = Add()([t, kv])
ps3 = Dense(1, activation='sigmoid')(t)
ps4 = Dense(1, activation='sigmoid')(t)

subject2_model = Model([t1_in, t2_in, k1_in, k2_in], [ps3, ps4])

k3v = position_embedding(Lambda(position_id)([t, k3]))
k4v = position_embedding(Lambda(position_id)([t, k4]))
kv = Average()([k3v, k4v])
t = Add()([t, kv])
po1 = Dense(num_classes, activation='sigmoid')(t)
po2 = Dense(num_classes, activation='sigmoid')(t)

object_model = Model([t1_in, t2_in, k1_in, k2_in, k3_in, k4_in], [po1, po2])

ok3v = position_embedding(Lambda(position_id)([t, ok1]))
ok4v = position_embedding(Lambda(position_id)([t, ok2]))
kv = Average()([ok3v, ok4v])
t = Add()([t, kv])

po3 = Dense(num_classes, activation='sigmoid')(t)
po4 = Dense(num_classes, activation='sigmoid')(t)

object2_model = Model([t1_in, t2_in, k1_in, k2_in, k3_in, k4_in, ok1_in, ok2_in], [po3, po4])

train_model = Model(
    [t1_in, t2_in, s1_in, s2_in, s3_in, s4_in, k1_in, k2_in, k3_in, k4_in, o1_in, o2_in, o3_in, o4_in, ok1_in, ok2_in],
    [ps1, ps2, ps3, ps4, po1, po2, po3, po4])

s1 = K.expand_dims(s1, 2)
s2 = K.expand_dims(s2, 2)
s3 = K.expand_dims(s3, 2)
s4 = K.expand_dims(s4, 2)

s1_loss = K.binary_crossentropy(s1, ps1)
s1_loss = K.sum(s1_loss * mask) / K.sum(mask)
s2_loss = K.binary_crossentropy(s2, ps2)
s2_loss = K.sum(s2_loss * mask) / K.sum(mask)
s3_loss = K.binary_crossentropy(s3, ps3)
s3_loss = K.sum(s3_loss * mask) / K.sum(mask)
s4_loss = K.binary_crossentropy(s4, ps4)
s4_loss = K.sum(s4_loss * mask) / K.sum(mask)

o1_loss = K.sum(K.binary_crossentropy(o1, po1), 2, keepdims=True)
o1_loss = K.sum(o1_loss * mask) / K.sum(mask)
o2_loss = K.sum(K.binary_crossentropy(o2, po2), 2, keepdims=True)
o2_loss = K.sum(o2_loss * mask) / K.sum(mask)
o3_loss = K.sum(K.binary_crossentropy(o3, po3), 2, keepdims=True)
o3_loss = K.sum(o3_loss * mask) / K.sum(mask)
o4_loss = K.sum(K.binary_crossentropy(o4, po4), 2, keepdims=True)
o4_loss = K.sum(o4_loss * mask) / K.sum(mask)

loss = (s1_loss + s2_loss + s3_loss + s4_loss) + (o1_loss + o2_loss + o3_loss + o4_loss)

train_model.add_loss(loss)
train_model.compile(optimizer=Adam(learning_rate))
train_model.summary()


def extract_items(text_in):
    R = []
    _tokens = tokenizer.tokenize(text_in)
    _t1, _t2 = tokenizer.encode(first=text_in)
    _t1, _t2 = np.array([_t1]), np.array([_t2])
    _k1, _k2 = subject_model.predict([_t1, _t2])
    _k1, _k2 = np.where(_k1[0] > 0.5)[0], np.where(_k2[0] > 0.4)[0]
    _subjects = []
    for i in _k1:
        j = _k2[_k2 >= i]
        if len(j) > 0:
            j = j[0]
            _subject = text_in[i - 1: j]
            _subjects.append((_subject, i, j))
    if _subjects:
        _subject2s = []
        _t1 = np.repeat(_t1, len(_subjects), 0)
        _t2 = np.repeat(_t2, len(_subjects), 0)
        _k1, _k2 = np.array([_s[1:] for _s in _subjects]).T.reshape((2, -1, 1))
        _k3, _k4 = subject2_model([_t1, _t2, _k1, _k2])
        for i, subject2 in enumerate(_subjects):
            _kk3, _kk4 = np.where(_k3[i] > 0.5)[0], np.where(_k4[i] > 0.4)[0]
            for _kkk3 in _kk3:
                for _kkk4 in _kk4:
                    if _kkk3 <= _kkk4:
                        _subject2 = text_in[_kkk3 - 1: _kkk4]
                        _subject2s.append((subject2[0], subject2[1], subject2[2], _subject2, _kkk3, _kkk4))

        if _subject2s:
            _t1, _t2 = tokenizer.encode(first=text_in)
            _t1, _t2 = np.array([_t1]), np.array([_t2])
            _t1 = np.repeat(_t1, len(_subject2s), 0)
            _t2 = np.repeat(_t2, len(_subject2s), 0)
            _objects = []
            _predicate = None
            _k1, _k2 = np.array([_a[1:3] for _a in _subject2s]).T.reshape((2, -1, 1))
            _k3, _k4 = np.array([_s[4:] for _s in _subject2s]).T.reshape((2, -1, 1))
            _o1, _o2 = object_model.predict([_t1, _t2, _k1, _k2, _k3, _k4])
            for si, ssubject in enumerate(_subject2s):
                _oo1, _oo2 = np.where(_o1[si] > 0.5), np.where(_o2[si] > 0.4)
                for _ooo1, _c1 in zip(*_oo1):
                    for _ooo2, _c2 in zip(*_oo2):
                        if _ooo1 <= _ooo2 and _c1 == _c2:
                            _object = text_in[_ooo1 - 1: _ooo2]
                            _predicate = id2predicate[_c2]
                            _objects.append((ssubject[0], ssubject[1], ssubject[2], ssubject[3], ssubject[4],
                                             ssubject[5], _predicate, _object, _ooo1, _ooo2))

            if _objects:
                _t1, _t2 = tokenizer.encode(first=text_in)
                _t1, _t2 = np.array([_t1]), np.array([_t2])
                _t1 = np.repeat(_t1, len(_objects), 0)
                _t2 = np.repeat(_t2, len(_objects), 0)
                _k1, _k2 = np.array([_a[1:3] for _a in _objects]).T.reshape((2, -1, 1))
                _k3, _k4 = np.array([_b[4:6] for _b in _objects]).T.reshape((2, -1, 1))
                _ok1, _ok2 = np.array([_s[8:] for _s in _objects]).T.reshape((2, -1, 1))
                _o3, _o4 = object2_model.predict([_t1, _t2, _k1, _k2, _k3, _k4, _ok1, _ok2])
                for n, oobject in enumerate(_objects):
                    _oo3, _oo4 = np.where(_o3[n] > 0.5), np.where(_o4[n] > 0.4)
                    for _ooo3, _c3 in zip(*_oo3):
                        for _ooo4, _c4 in zip(*_oo4):
                            if _ooo3 <= _ooo4 and _c3 == _c4:
                                _object2 = text_in[_ooo3 - 1: _ooo4]
                                R.append((oobject[0], oobject[3], oobject[6], oobject[7], _object2))
                                break
        spo_list = set()
        for s, ss, p, o, oo in R:
            spo_list.add((s, ss, p, o, oo))
        return list(spo_list)
    else:
        return []


class Evaluate(Callback):
    def __init__(self):
        self.F1 = []
        self.best = 0.
        self.passed = 0
        self.stage = 0

    def on_batch_begin(self, batch, logs=None):
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
            train_model.save_weights('./data/advance/best_model.weights')
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best))

    def evaluate(self):
        orders = ['subject', 'ssubject', 'predicate', 'object', 'oobject']
        A, B, C = 1e-10, 1e-10, 1e-10
        F = open('dev_pred.json', 'w')
        for d in dev_data:
            R = set(extract_items(d['text']))
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
        return 2 * A / (B + C), A / B, A / C


train_D = data_generator(train_data)
evaluator = Evaluate()

if __name__ == '__main__':
    train_model.fit_generator(train_D.__iter__(),
                              steps_per_epoch=1000,
                              epochs=300,
                              callbacks=[evaluator]
                              )
else:
    train_model.load_weights('./data/advance/best_model.weights')
