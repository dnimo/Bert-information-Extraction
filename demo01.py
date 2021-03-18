from keras_bert import bert
from keras_bert import load_trained_model_from_checkpoint,Tokenizer
import codecs

config_path = '../2021Bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../2021Bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../2021Bert/chinese_L-12_H-768_A-12/vocab.txt'

token_dict = {}
with codecs.open(dict_path,'r','utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
#重写tokenizer
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
             R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')
                # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')
                # 剩余的字符是UNK
        return  R

tokenizer = OurTokenizer(token_dict)

from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam

def seq_gather(x):
    # seq是[none,seq_len,s_size]的格式，idxs是[None,1]的格式
    # 在seq的第i个序列中选出第i个向量，最终输出[None,s_size]的向量
    seq,idxs = x
    idxs = K.cast(idxs,'int32')
    batch_idxs = K.arange(0,K.shape(seq)[0])
    batch_idxs = K.expand_dims(batch_idxs,1)
    idxs = K.concatenate([batch_idxs,idxs],1)
    return K.tf.gather_nd(seq,idxs)

bert_model = load_trained_model_from_checkpoint(config_path,checkpoint_path,seq_len=None)

for l in bert_model.layers:
    l.trainable = True

t1_in = Input(shape=(None,))
