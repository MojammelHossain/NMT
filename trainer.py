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

from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from language import Lang
from decoder import *
from encoder import *
from evaluation import *
from utils import save_model
from plot import plot
from bleu import bleu_score

plt.switch_backend('agg')


class TrainModel:
  """train encoder-decoder architecture
  Attributes:
    max_length (int) : sequence max length
    sos_token (int) : sos_token from language.py file
    learning_rate (float) : determine the steps of optimizer Default : 0.001
    batch_size (int) : number of sequence to load together in model Default : 1
    epochs (int) : number of epochs Default : 1
    path (str) : path for saving model Default : 'C:/'
  """

  def __init__(self, *arg, **kargs):

    self.max_length = arg[0]
    self.sos_token = arg[1]
    self.learning_rate = kargs['learning_rate'] if 'learning_rate' in kargs.keys() else 0.001
    self.batch_size = kargs['batch_size'] if 'batch_size' in kargs.keys() else 1
    self.epochs = kargs['epochs'] if 'epochs' in kargs.keys() else 1
    self.path = kargs['path'] if 'path' in kargs.keys() else 'C:/'
    self.snapshot = kargs['snapshot'] if 'snapshot' in kargs.keys() else None
    self.criterion = nn.NLLLoss()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Model will train in {}'.format(self.device))


  def train(self, input_tensor, target_tensor, encoder, decoder, \
            encoder_optimizer, decoder_optimizer):

    encoder.train()
    decoder.train()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_hidden = encoder.init_hidden(input_tensor.shape[0], self.device)
    input_tensor = input_tensor.squeeze(1).to(self.device)
    target_tensor = target_tensor.squeeze(1)
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
    
    decoder_inputs = torch.tensor([[self.sos_token]], device=self.device\
                                 ).new_full((target_tensor.shape[0], 1),\
                                  self.sos_token)

    loss = 0
    target = target_tensor.T
    for i in range(target_tensor.shape[1]):
      decoder_output, decoder_hidden = decoder(decoder_inputs, decoder_hidden)
      topv, topi = decoder_output.topk(1)
      decoder_input = topi.squeeze().detach()
      decoder_input = decoder_input.view(-1,1)

      loss += self.criterion(decoder_output, target[i].to(self.device))

    loss = loss/target_tensor.shape[1]

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


  def train_batches(self, encoder, encoder_optimizer, decoder, \
                    decoder_optimizer, data_loader):

    total_loss = 0


    for i, batch in enumerate(data_loader):
      inputs = batch.gather(1, torch.zeros([batch.shape[0], 1, \
                                            batch.shape[2]]).long())
      targets = batch.gather(1, torch.ones([batch.shape[0], 1, \
                                            batch.shape[2]]).long())
      total_loss += self.train(inputs, targets, encoder, decoder, \
                               encoder_optimizer, decoder_optimizer)
    return total_loss


  def train_epochs(self, encoder, decoder, train_data, dev_data=None, \
                   pred_=None):

    encoder.to(self.device)
    decoder.to(self.device)

    train_dataloader = DataLoader(train_data, self.batch_size, shuffle=True)

    encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, \
                                   encoder.parameters()), lr=self.learning_rate)
    decoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, \
                                   decoder.parameters()), lr=self.learning_rate)
    checkpoint_epoch=0
    total_epoch = 0
    train_score=[0]
    val_score=[0]

    if self.snapshot != None:
      print('loading model')
      checkpoint = torch.load(self.snapshot)
      total_epoch = checkpoint['epochs']
      encoder.load_state_dict(checkpoint['encoder'])
      decoder.load_state_dict(checkpoint['decoder'])
      encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
      decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
    
    t = tqdm(range(1, self.epochs+1))
    try:
        for epoch in t:

            loss = self.train_batches(encoder, encoder_optimizer, decoder, \
                                      decoder_optimizer, train_dataloader)
            if (epoch) % 10 == 0:
              t.set_description("loss %s" % loss)
              checkpoint_epoch += 10
              train_predict = pred_.predict_dataset(encoder, decoder, train_data)
              if dev_data is None:
                print('Please provide a dev set for visualize the plot')
              else:
                dev_predict = pred_.predict_dataset(encoder, decoder, dev_data)
                train_score.append(bleu_score(train_predict[1], train_predict[0]))
                val_score.append(bleu_score(dev_predict[1], dev_predict[0]))
                plot(train_scor=train_score, val_scor=val_score, epoch=checkpoint_epoch)
              
              path = self.path + "/model" + str(total_epoch+epoch) +".pt"
              save_model(checkpoint_epoch+total_epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, path)
    except KeyboardInterrupt:
        # Code to "save"
        print('save model')
        save_model(checkpoint_epoch+total_epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, path)
    path = self.path+"/model" + str(total_epoch+self.epochs) +".pt"
    save_model(total_epoch+self.epochs, encoder, decoder, encoder_optimizer, decoder_optimizer, path)