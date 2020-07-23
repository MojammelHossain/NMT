"""Copyright"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Base enocder class
    Attributes:
        input_size (int) : number of word in input language
        hidden_size (int) : number of feature
        padding_idx (int) : if given initialize zeros Default : None
        num_layers (int) : number of RNN Default '1'
        rnn (str) : RNN architecture to follow Default : 'LSTM'
        bidirectional (bool) : If 'True' become a bidirectional RNN
        architecture Default 'False'
        dropout (float) : if provide then add a Dropout layerto each
        RNN layer outputs except last layer Default '0' value provide
        between (0-1)
    """
    def __init__(self, *arg, padding_idx=None, num_layers=1, rnn='LSTM',\
                 bidirectional=False, dropout=0.):
        super(Encoder, self).__init__()

        self.input_size = arg[0]
        self.hidden_size = arg[1]
        self.padding_idx = padding_idx
        self.num_layers = num_layers
        self.rnn = rnn
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.embedding = nn.Embedding(self.input_size, self.hidden_size,\
                                          padding_idx=self.padding_idx)
        if self.rnn == 'LSTM':
            self.archi = nn.LSTM(self.hidden_size, self.hidden_size,\
                                 dropout=self.dropout, batch_first=True,\
                                 bidirectional=self.bidirectional)
        else:
            self.archi = nn.GRU(self.hidden_size, self.hidden_size,\
                                dropout=self.dropout, batch_first=True,\
                                bidirectional=self.bidirectional)


    def load_weights(self, weights_matrix, requires_grad=False):
        """load pretrained words vectors"""

        self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))
        self.embedding.weight.requires_grad = requires_grad


    def forward(self, inputs, hidden):
        """one forward pass through encoder network"""

        embedded = self.embedding(inputs)
        output, hidden = self.archi(embedded, hidden)

        return output, hidden


    def init_hidden(self, batch_size=1, device='cpu'):
        """initialize hidden states"""

        direction = 2 if self.bidirectional else 1
        if self.rnn == 'LSTM':
            return (torch.zeros(self.num_layers*direction, batch_size,\
                                self.hidden_size, device=device),

                    torch.zeros(self.num_layers*direction, batch_size,\
                                self.hidden_size, device=device))
        return  torch.zeros(self.num_layers*direction, batch_size,\
                                self.hidden_size, device=device)
