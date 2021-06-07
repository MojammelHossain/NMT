import string
import re
import copy
import torch
import spacy
from encoder import Encoder
from decoder import Decoder
from language import Lang
from utils import Helper
from trainer import TrainModel
from evaluation import Predict
from string import punctuation


MAX_LENGTH = 10
# add missing bangla punctuation
punctuation = punctuation + '।'
def ben_tokenizer(sentence):
    sentence = text = sentence.translate(str.maketrans({key: " {0} ".format(key) for key in punctuation}))
    sentence = re.sub(' +', ' ', sentence)
    list_ = sentence.strip().split(' ')
    list_ = [i for i in list_ if i] 
    return list_

if __name__ == "__main__":
    helper = Helper(lang1='eng', lang2='ben', reverse=True, max_length=10)
    eng_tokenizer = spacy.load("en_core_web_sm")

    train_data = helper.read_langs('/content/drive/My Drive/CSE299/en_bn_corpus/supara_en_bn.txt')
    dev_data = helper.read_langs('/content/drive/My Drive/CSE299/en_bn_corpus/tatoeba_en_bn.txt')
    train_data = helper.filter_pairs(train_data)
    dev_data = helper.filter_pairs(dev_data)

    for i, pair in enumerate(train_data):
        train_data[i][0] = ben_tokenizer(pair[0])
        train_data[i][1] = [token.text for token in eng_tokenizer(pair[1].lower())]
    for i, pair in enumerate(dev_data):
        dev_data[i][0] = ben_tokenizer(pair[0])
        dev_data[i][1] = [token.text for token in eng_tokenizer(pair[1].lower())]

    input_lang, output_lang, pairs, train_data = helper.prepare_data(train_data)


    dev_data = helper.filter_pairs(dev_data)
    
    for i, _ in enumerate(dev_data):
      tensors = helper.tensors_from_pair(dev_data[i])
      dev_data[i][0] = tensors[0]
      dev_data[i][1] = tensors[1]
    dev_data[5]

    train_data = helper.padding(train_data, input_lang.pad_token)
    dev_data = helper.padding(dev_data, input_lang.pad_token)

    encoder = Encoder(input_lang.n_words, 300, padding_idx=input_lang.pad_token, \
                  bidirectional=True)
    decoder = Decoder(output_lang.n_words, 300, padding_idx=output_lang.pad_token,\
                  bidirectional=True)

    pred_= Predict(MAX_LENGTH, input_lang, output_lang, helper, 8000)
    train = TrainModel(MAX_LENGTH, input_lang.sos_token, batch_size=8000, epochs=20, learning_rate=0.0001, path="/content/snap", snapshot='/content/snap/model10.pt')
    train.train_epochs(encoder,decoder, train_data, dev_data, pred_)