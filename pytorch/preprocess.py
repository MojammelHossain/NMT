"""Copyright"""
import copy
import numpy as np
from .language import Lang

class utils:
    """helper class for reading data and create word bank"""

    def __init__(self):
        """initialize class object"""
    def read_langs(self, lang1, lang2, path, reverse=False):
        """read data file from given path and create Lang objects"""

        print("Reading lines...")

        # Read the file and split into lines
        lines = open(path, encoding='utf-8').read().strip().split('\n')

        # Split every line into pairs
        pairs = [[s for s in l.split('\t')] for l in lines]

        # Reverse pairs, make Lang instances
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang(lang2)
            output_lang = Lang(lang1)
        else:
            input_lang = Lang(lang1)
            output_lang = Lang(lang2)

        return input_lang, output_lang, pairs


    def filter_pair(self, pair, max_length):
        """choose pair based on max_length"""

        return len(pair[0].split(' ')) < max_length and \
            len(pair[1].split(' ')) < max_length


    def filter_pairs(self, pairs, max_length):
        """choose pairs"""

        return [pair for pair in pairs if self.filter_pair(pair, max_length)]


    def indexes_from_sentence(self, lang, sentence):
        """convert sentence into its corresponding indices"""

        return [lang.word2index[word] if word in lang.word2index.keys()\
                  else lang.unk_token for word in sentence.split(' ')]


    def tensor_from_sentence(self, lang, sentence):
        """convert sentence indices into numpy array"""

        indices = self.indexes_from_sentence(lang, sentence)
        indices.append(lang.eos_token)
        return np.array(indices)


    def tensors_from_pair(self, input_lang, output_lang, pair):
        """convert pair into indices"""

        input_tensor = self.tensor_from_sentence(input_lang, pair[0])
        target_tensor = self.tensor_from_sentence(output_lang, pair[1])
        return (input_tensor, target_tensor)


    def prepare_data(self, lang1, lang2, max_length, path, reverse=False):
        """prepare data for model"""

        input_lang, output_lang, pairs = self.read_langs(lang1, lang2, path, \
                                                        reverse)
        print("Read %s sentence pairs" % len(pairs))

        pairs = self.filter_pairs(pairs, max_length)
        print("Trimmed to %s sentence pairs" % len(pairs))

        print("Counting words...")
        for pair in pairs:
            input_lang.add_sentence(pair[0])
            output_lang.add_sentence(pair[1])

        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)

        # remove words with occurrence less than 6
        input_lang.most_common_words(5)
        output_lang.most_common_words(5)
        print("Most common words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)

        pair_tensors = copy.deepcopy(pairs)
        for i, _ in enumerate(pair_tensors):
            tensors = self.tensors_from_pair(input_lang, output_lang, pair_tensors[i])
            pair_tensors[i][0] = tensors[0]
            pair_tensors[i][1] = tensors[1]

        return input_lang, output_lang, pairs, pair_tensors


    def padding_sentence(self, word_indices, max_length, pad_token):
        """senctence -> fixed length word vector"""

        if max_length > len(word_indices):
            word_indices = np.concatenate([word_indices, np.array([pad_token \
                              for _ in range(max_length - len(word_indices))])])
        return word_indices


    def padding(self, pair_tensors, max_length, pad_token):
        """pairs -> tensors"""

        for i, _ in enumerate(pair_tensors):
            pair_tensors[i][0] = self.padding_sentence(pair_tensors[i][0], \
                                                       max_length, pad_token)
            pair_tensors[i][1] = self.padding_sentence(pair_tensors[i][1], \
                                                       max_length, pad_token)
        return np.array(pair_tensors)
