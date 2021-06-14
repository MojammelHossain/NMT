import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """Dase decoder class

    Attributes:
        output_size (int) : number of word in output language
        hidden_size (int) : number of feature
        padding_idx (int) : if given initialize zeros Default : 1
        rnn (str) : RNN architecture to follow Default : 'LSTM'
        bidirectional (bool) : if input pass from a bidirectional encoder
         then True Default : False

    """
    def __init__(self, *arg, padding_idx=None, rnn='LSTM', bidirectional=False):

        super(Decoder, self).__init__()

        self.output_size = arg[0]
        self.hidden_size = arg[1]
        self.padding_idx = padding_idx
        self.rnn = rnn
        self.direction = 2 if bidirectional else 1

        self.embedding = nn.Embedding(self.output_size, self.hidden_size, \
                                    padding_idx=self.padding_idx)

        if self.rnn == 'LSTM':
            self.archi = nn.LSTM(self.hidden_size, \
                                 self.hidden_size*self.direction, \
                                 batch_first=True)
        else:
            self.archi = nn.GRU(self.hidden_size, \
                                self.hidden_size*self.direction, \
                                batch_first=True)

        self.out = nn.Linear(self.hidden_size*self.direction, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def load_embed_weights(self, weights_matrix, requires_grad=False):
        """load pretrained word vectors"""
        self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))
        self.embedding.weight.requires_grad = requires_grad

    def forward(self, inputs, hidden):
        """one forward pass through decoder network"""

        output = self.embedding(inputs)
        output = F.relu(output)

        output, hidden = self.archi(output, hidden)
        output = self.out(output).view(len(inputs), -1)
        output = self.softmax(output)

        return output, hidden

    def init_hidden(self, batch_size=1, device='cpu'):
        """initialize hidden state"""
        if self.rnn == 'LSTM':
            return (torch.zeros(1, batch_size, self.hidden_size*self.direction,\
              device=device), torch.zeros(1, batch_size, \
              self.hidden_size*self.direction, device=device))

        return torch.zeros(1, batch_size, self.hidden_size*self.direction, \
                       device=device)


class AttnDecoderRNN(Decoder):
    """Attention decoder class 
    *note : check decoder class first
    Attributes:
      dropout_p (float) : adding a dropout layer after embedding layer 
      Default = 0.1
      max_length (int) : max sequence length
    """

    def __init__(self, *arg, max_length, padding_idx=None, \
               rnn='LSTM', bidirectional=False, dropout_p=0.1):

        Decoder.__init__(*arg, padding_idx, rnn, bidirectional)

        self.dropout_p = dropout_p
        self.max_length = max_length

        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, inputs, hidden, encoder_outputs):
        """one forward pass through attention network"""

        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]),\
                                    1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs)

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)

        output, hidden = self.archi(output, hidden)
        output = self.out(output[0])
        output = self.softmax(output)

        return output, hidden, attn_weights
