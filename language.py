class Lang:
    """Base class for create language wordbook

    Attributes:
        name (str)        : language name
        word2index (dict) : provide index of a corresponding word
        index2word (dict) : provide word of a corresponding index
        word2count (dict) : count number of words occurrence in corpus
        n_words (int)     : number of unique words

    """
    def __init__(self, name):

        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<sos>", 1: "<eos>", 2: "<unk>", 3: "<pad>"}
        self.n_words = 4  # Count first 4 word (0-3)
        self.sos_token = 0
        self.eos_token = 1
        self.unk_token = 2
        self.pad_token = 3


    def add_sentence(self, sentence):
        """split sentence into words"""

        for word in sentence:
            self.add_word(word)


    def add_word(self, word):
        """add word into word bank"""

        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


    def most_common_words(self, occurrence):
        """remove rare words based on occurrence (int) in corpus"""

        self.word2index = {}
        self.index2word = {0: "<sos>", 1: "<eos>", 2: "<unk>", 3: "<pad>"}
        self.n_words = 4

        for word in self.word2count.keys():
            if self.word2count[word] > 0:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1
