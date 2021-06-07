import time
import sys
import logging

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


def plot_score(train_score, val_score, interval):

  plt.clf()
  plt.plot(interval, train_score, label='Train')
  plt.plot(interval, val_score, label='Validation', color='green')
  plt.xlabel('Epochs')
  plt.ylabel('BLEU Score')
  plt.title('Train and validation BLEU score curves')
  plt.xticks(interval)
  plt.legend()
  plt.show()


def plot(**kargs):

  try:
    train_score = kargs['train_scor']
    val_score = kargs['val_scor']
    epoch = kargs['epoch']
  except KeyError:
    print("Please provide the all necessary arguments")
    sys.exit(1)
  interval = np.arange(0, epoch+1, 10)
  clear_output()
  plot_score(train_score, val_score, interval)