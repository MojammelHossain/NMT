import string
import re
import copy
import torch
import spacy
import configparser
from utils import *
from language import Lang
from encoder import Encoder
from decoder import Decoder
from trainer import TrainModel
from evaluation import Predict
from string import punctuation


# add missing bangla punctuation
punctuation = punctuation + '।'
def ben_tokenizer(sentence):
    sentence = text = sentence.translate(str.maketrans({key: " {0} ".format(key) for key in punctuation}))
    sentence = re.sub(' +', ' ', sentence)
    list_ = sentence.strip().split(' ')
    list_ = [i for i in list_ if i] 
    return list_

if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read('experiment.ini')
    config = initialize_env(dict(config['DEFAULT']))
    helper = Helper(lang1=config['lang1'], lang2=config['lang2'], reverse=config['reverse'], max_length=config['max_length'])
    eng_tokenizer = spacy.load("en_core_web_sm")

    train_data = helper.read_langs(config['train_path'])
    dev_data = helper.read_langs(config['eval_path'])
    train_data = helper.filter_pairs(train_data)
    dev_data = helper.filter_pairs(dev_data)

    for i, pair in enumerate(train_data):
        train_data[i][0] = ben_tokenizer(pair[0])
        train_data[i][1] = [token.text for token in eng_tokenizer(pair[1].lower())]
    for i, pair in enumerate(dev_data):
        dev_data[i][0] = ben_tokenizer(pair[0])
        dev_data[i][1] = [token.text for token in eng_tokenizer(pair[1].lower())]

    
    input_lang, output_lang, pairs, train_data = helper.prepare_data(train_data, load=config['obj_path'])


    dev_data = helper.filter_pairs(dev_data)
    
    for i, _ in enumerate(dev_data):
      tensors = helper.tensors_from_pair(dev_data[i])
      dev_data[i][0] = tensors[0]
      dev_data[i][1] = tensors[1]
    dev_data[5]

    train_data = helper.padding(train_data, input_lang.pad_token)
    dev_data = helper.padding(dev_data, input_lang.pad_token)

    encoder = Encoder(input_lang.n_words, config['hidden_size'], padding_idx=input_lang.pad_token, \
                  bidirectional=config['bidirectional'])
    decoder = Decoder(output_lang.n_words, config['hidden_size'], padding_idx=output_lang.pad_token,\
                  bidirectional=config['bidirectional'])

    if config['obj_path'] == None:
        print("Saving language objects: {}/".format(config['root']))
        helper.save_lang_object(config['root'])
    pred_= Predict(config['max_length'], input_lang, output_lang, helper, config['batch_size'])
    train = TrainModel(config['max_length'], input_lang.sos_token, batch_size=config['batch_size'], epochs=config['epochs'], learning_rate=config['learning_rate'], path=config['root'], snapshot=config['checkpoint'], eval_frequency=config['eval_frequency'])
    train.train_epochs(encoder,decoder, train_data, dev_data, pred_)