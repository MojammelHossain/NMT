# Neural Machine Translation with RNN variants.

## Introduction
Encoder-Decoder architecture for neural machine translation `BN->EN`.
* Both LSTM and GRU can be used as Encoder or Decoder.
* Scratch implementation of vocab `language.py` and trainer `trainer.py` class.
* Implementation of [Bahdanau attention decoder](https://arxiv.org/pdf/1409.0473.pdf).

## Setup
Install the following if not installed.
* python 3.x
* torch cuda version
* scarceblue
* configparser

## Training
* Keep your preprocess data in the `data` folder check out the `sample.txt` for data format.
* Change the appropriate variable inside `experiment.ini` i.e. `lang1, rnn, hidden_size`.
* `reverse` a `bool` variable will change the model training i.e. `BN->EN` to `EN->BN`
* Training: `python train.py`

### Resumming Training
`language.py` and trainer `trainer.py` class.
