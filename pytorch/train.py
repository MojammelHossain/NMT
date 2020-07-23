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
from .decoder import Decoder, AttnDecoderRNN
from .encoder import Encoder

plt.switch_backend('agg')


class TrainModel:
  """train encoder-decoder architecture
  Attributes:
    max_length (int) : sequence max length
    sos_token (int) : sos_token from language.py file
    eos_token (int) : eos_token from language.py file
    learning_rate (float) : determine the steps of optimizer Default : 0.001
    batch_size (int) : number of sequence to load together in model Default : 1
    epochs (int) : number of epochs Default : 1
    path (str) : path for saving model Default : 'C:/'
  """

  def __init__(self, *arg, learning_rate=0.001, batch_size=1, epochs=1, path='C:/'):

    self.max_length = arg[0]
    self.sos_token = arg[1]
    self.eos_token = arg[2]
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.epochs = epochs
    self.path = path
    self.criterion = nn.NLLLoss()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


  def train(input_tensor, target_tensor, enocder, decoder, encoder_optimizer, decoder_optimizer):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    enocder_hidden = encoder.init_hidden(input_tensor.shape[0], self.device)
    input_tensor = input_tensor.squeeze(1).to(self.device)
    target_tensor = target_tensor.squeeze(1)
    encoder_out, enocder_hidden = enocder(input_tensor, enocder_hidden)

    if encoder.bidirectional:
      if enocder.rnn == 'LSTM':
        decoder_hidden = (torch.cat((encoder_hidden[0][0], \
                                    encoder_hidden[1][0]),1).unsqueeze(0),
                          torch.cat((encoder_hidden[0][1], \
                                  encoder_hidden[1][1]),1).unsqueeze(0))
      else:
        decoder_hidden = torch.cat((encoder[0], eoncder[1]), 1)
    else:
      decoder_hidden = encoder_hidden
    
    decoder_inputs = torch.tensor([[self.sos_token]], device=device).new_full(\
                                  (target_tensor.shape[0], 1), self.sos_token)

    output = torch.zeros(len(target_tensor), 1, decoder.output_size)
    for i in range(len(target_tensor.shape[1])):
      decoder_output, decoder_hidden = decoder(decoder_inputs, decoder_hidden)
      topv, topi = decoder_output.topk(1)
      decoder_input = topi.squeeze().detach()
      decoder_input = decoder_input.view(-1,1)

      if i==0:
        output = decoder_output.unsqueeze(1)
      else:
        output = torch.cat((x,y), axis=1)

    loss = self.criterion(output, target_tensor)

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


  def train_batches(encoder, encoder_optimizer, decoder, decoder_optimizer,\
                    data_loader):

    total_loss = 0

    enocder.to(self.device)
    decoder.to(self.device)

    for i, batch in enumerate(data_loader):
      inputs = batch.gather(1, torch.zeros([batch.shape[0], 1, \
                                            batch.shape[2]]).long())
      targets = batch.gather(1, torch.ones([batch.shape[0], 1, \
                                            batch.shape[2]]).long())
      loss = train(inputs, targets, encoder, decoder, \
                  encoder_optimizer, decoder_optimizer)


  def train_epochs(self, enocder, decoder, train_data, dev_data=None):

    train_dataloader = DataLoader(train_data, self.batch_size)
    if dev_data != None:
      dev_dataloader = DataLoader(dev_data, self.batch_size)
    
    encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, \
                                   encoder1.parameters()), lr=learning_rate)
    decoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, \
                                   decoder.parameters()), lr=learning_rate)
    
    for epoch in self.epochs:
      self.train_batches(encoder, encoder_optimizer, decoder, \
                         decoder_optimizer, train_dataloader)