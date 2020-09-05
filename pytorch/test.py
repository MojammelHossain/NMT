import logging
import math
import time
import pickle
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from torch import optim
from torch.utils.data import DataLoader
from .language import Lang
from .decoder import *
from .encoder import *
from tqdm import tqdm

plt.switch_backend('agg')


class Predict:
  """
  Attributes:
    
  """

  def __init__(self, max_length, input_vocab, output_vocab, helper, batch_size=1):

    self.max_length = max_length
    self.input_vocab = input_vocab
    self.output_vocab = output_vocab
    self.batch_size = batch_size
    self.util = helper
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


  def predict(self, input_tensor, encoder, decoder):

    encoder.eval()
    decoder.eval()
    batch_shape = input_tensor.shape[0]

    encoder_hidden = encoder.init_hidden(batch_shape, self.device)
    input_tensor = input_tensor.squeeze(1).to(self.device)
    encoder_out, encoder_hidden = encoder(input_tensor, encoder_hidden)

    if encoder.bidirectional:
      if encoder.rnn == 'LSTM':
        decoder_hidden = (torch.cat((encoder_hidden[0][0], \
                                    encoder_hidden[1][0]),1).unsqueeze(0),
                          torch.cat((encoder_hidden[0][1], \
                                  encoder_hidden[1][1]),1).unsqueeze(0))
      else:
        decoder_hidden = torch.cat((encoder_hidden[0], \
                                    encoder_hidden[1]), 1).unsqueeze(0)
    else:
      decoder_hidden = encoder_hidden
    
    decoder_inputs = torch.tensor([[self.input_vocab.sos_token]], \
                            device=self.device).new_full((batch_shape, 1), \
                            self.input_vocab.sos_token)

    pred = torch.zeros(self.max_length, batch_shape)

    for i in range(self.max_length):
      decoder_output, decoder_hidden = decoder(decoder_inputs, decoder_hidden)
      topv, topi = decoder_output.topk(1)
      decoder_input = topi.squeeze().detach()
      decoder_input = decoder_input.view(-1,1)
      pred[i] = topi.view(1, -1)

    return pred


  def predict_batches(self, encoder, decoder, data_loader):

    predicts = []
    tars = []

    for i, batch in enumerate(data_loader):
      inputs = batch.gather(1, torch.zeros([batch.shape[0], 1, \
                                            batch.shape[2]]).long())
      targets = batch.gather(1, torch.ones([batch.shape[0], 1, \
                                            batch.shape[2]]).long())
      pred = self.predict(inputs, encoder, decoder)

      tars.append(targets.squeeze())
      predicts.append(pred.T)
    return predicts, tars


  def predict_dataset(self, encoder, decoder, data):

    encoder.to(self.device)
    decoder.to(self.device)

    dataloader = DataLoader(data, self.batch_size, shuffle=True)
    predicts, tars = self.predict_batches(encoder, decoder, dataloader)
    predicts, tars = self.tensor_to_sentence(predicts, tars)
    return (predicts, tars)
  
  def tensor_to_sentence(self, predicts, targets):
    pred_sentences = []
    tars_sentences = []
    for i in range(len(predicts)):
      pred = np.array(predicts[i], dtype=np.int32)
      tars = np.array(targets[i], dtype=np.int32)
      for j in range(pred.shape[0]):
        pred_sentence = []
        tars_sentence = []
        for k in range(len(pred[j])):
          pred_sentence.append(self.output_vocab.index2word[pred[j][k]])
          tars_sentence.append(self.output_vocab.index2word[tars[j][k]])
        pred_sentences.append(' '.join(pred_sentence))
        tars_sentences.append(' '.join(tars_sentence))
    return pred_sentences, tars_sentences


  def predict_sentence(self, encoder, decoder, data):

    encoder.to(self.device)
    decoder.to(self.device)

    data = self.util.tensor_from_sentence(self.input_vocab, data)
    data = self.util.padding_sentence(data, self.input_vocab.pad_token)
    data = torch.from_numpy(data).long().view(1,1,-1)
    predict_tensor = self.predict(data, encoder, decoder).view(1, -1)

    return predict_tensor