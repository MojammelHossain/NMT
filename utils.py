import os
import copy
import torch
import pickle
import numpy as np
from language import Lang


def initialize_env(config):
    try:
        os.mkdir(config['root'])
    except:
        print("Directory found!!!!")

    config['batch_size'] = int(config['batch_size'])
    config['bidirectional'] = True if config['bidirectional'] == 'true' else False
    config['checkpoint'] = None if config['checkpoint'] == 'false' else config['checkpoint']
    config['reverse'] = True if config['reverse'] == 'true' else False
    config['rnn'] = 'LSTM' if config['rnn'] == 'lstm' else 'GRU'
    config['max_length'] = int(config['max_length'])
    config['epochs'] = int(config['epochs'])
    config['learning_rate'] = float(config['learning_rate'])
    config['hidden_size'] = int(config['hidden_size'])
    config['eval_frequency'] = int(config['eval_frequency'])
    config['obj_path'] = None if config['obj_path'] == 'false' else config['obj_path']
    return config

def save_model(epochs, encoder, decoder, encoder_optimizer, decoder_optimizer, path):
    torch.save({
            'epochs': epochs,
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'encoder_optimizer': encoder_optimizer.state_dict(),
            'decoder_optimizer': decoder_optimizer.state_dict()
          }, path)

class Helper:
    """helper class for reading data and create word bank"""

    def __init__(self, **kwargs):
        """initialize class object"""
        self.lang1_name = kwargs['lang1']
        self.lang2_name = kwargs['lang2']

        try:
          self.max_length = kwargs['max_length']
          self.reverse = kwargs['reverse']
        except:
          self.max_length = 0
          self.reverse = False

        if self.reverse:
            self.input_lang = Lang(self.lang2_name)
            self.output_lang = Lang(self.lang1_name)
        else:
            self.input_lang = Lang(self.lang1_name)
            self.output_lang = Lang(self.lang2_name)
        

    def read_langs(self, path):
        """read data file from given path and create Lang objects"""

        print("Reading lines...")

        # Read the file and split into lines
        lines = open(path, encoding='utf-8').read().strip().split('\n')

        # Split every line into pairs
        pairs = [[s for s in l.split('\t')] for l in lines]
        print("Total %s sentence pairs\n" % len(pairs))


        # Reverse pairs, make Lang instances
        if self.reverse:
            pairs = [list(reversed(p)) for p in pairs]

        return pairs


    def filter_pair(self, pair, lower=None):
        """choose pair based on max_length"""
        if lower is not None:
          if isinstance(pair[0], list):
            return len(pair[0]) > lower and len(pair[1]) > lower and \
                   len(pair[0]) < self.max_length and \
                   len(pair[1]) < self.max_length
        if isinstance(pair[0], list):
          return len(pair[0]) < self.max_length and \
                 len(pair[1]) < self.max_length
        return len(pair[0].split(' ')) < self.max_length and \
               len(pair[1].split(' ')) < self.max_length


    def filter_pairs(self, pairs, lower = None):
        """choose pairs"""

        return [pair for pair in pairs if self.filter_pair(pair, lower)]


    def indexes_from_sentence(self, lang, sentence):
        """convert sentence into its corresponding indices"""

        return [lang.word2index[word] if word in lang.word2index.keys()\
                  else lang.unk_token for word in sentence]


    def tensor_from_sentence(self, lang, sentence):
        """convert sentence indices into numpy array"""

        indices = self.indexes_from_sentence(lang, sentence)
        indices.append(lang.eos_token)
        return np.array(indices)


    def tensors_from_pair(self, pair):
        """convert pair into indices"""

        input_tensor = self.tensor_from_sentence(self.input_lang, pair[0])
        target_tensor = self.tensor_from_sentence(self.output_lang, pair[1])
        return (input_tensor, target_tensor)


    def prepare_data(self, pairs, occurence=None, load=None):
        """prepare data for model"""

        pairs = self.filter_pairs(pairs)
        print("Trimmed to %s sentence pairs as per max_length\n" % len(pairs))

        if load == None:
            for pair in pairs:
                self.input_lang.add_sentence(pair[0])
                self.output_lang.add_sentence(pair[1])

            if occurence != None:
                self.input_lang.most_common_words(5)
                self.output_lang.most_common_words(5)
                print("Most common words:")
                print(self.input_lang.name, self.input_lang.n_words)
                print(self.output_lang.name, self.output_lang.n_words)
        else:
            self.load_lang_object(load)

        print("Counted words:")
        print(self.input_lang.name, self.input_lang.n_words)
        print(self.output_lang.name, self.output_lang.n_words)

        pair_tensors = copy.deepcopy(pairs)
        for i, _ in enumerate(pair_tensors):
            tensors = self.tensors_from_pair(pair_tensors[i])
            pair_tensors[i][0] = tensors[0]
            pair_tensors[i][1] = tensors[1]

        return self.input_lang, self.output_lang, pairs, pair_tensors


    def padding_sentence(self, word_indices, pad_token):
        """senctence -> fixed length word vector"""

        if self.max_length > len(word_indices):
            word_indices = np.concatenate([word_indices, np.array([pad_token \
                              for _ in range(self.max_length - len(word_indices))])])
        return word_indices


    def padding(self, pair_tensors, pad_token):
        """pairs -> tensors"""

        for i, _ in enumerate(pair_tensors):
            pair_tensors[i][0] = self.padding_sentence(pair_tensors[i][0], \
                                                       pad_token)
            pair_tensors[i][1] = self.padding_sentence(pair_tensors[i][1], \
                                                       pad_token)
        return np.array(pair_tensors)

    def save_lang_object(self, path):
        with open((path+'/input_lang.pkl'), 'wb') as output:
            pickle.dump(self.input_lang, output, pickle.HIGHEST_PROTOCOL)
        with open((path+'/output_lang.pkl'), 'wb') as output:
            pickle.dump(self.output_lang, output, pickle.HIGHEST_PROTOCOL)

    def load_lang_object(self, path):
        print("Loading language objects from : {}/".format(path))
        with open((path+'/input_lang.pkl'), 'rb') as input:
            self.input_lang = pickle.load(input)
        with open((path+'/output_lang.pkl'), 'rb') as input:
            self.output_lang = pickle.load(input)