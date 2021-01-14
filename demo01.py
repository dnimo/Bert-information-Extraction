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

print(tokenizer.tokenize('今天的  天气不错！'))
