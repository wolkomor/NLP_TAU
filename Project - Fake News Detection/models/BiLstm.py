import torch
import argparse
import os
import torch.nn.functional as F
import random
import time
from sklearn.metrics import precision_recall_fscore_support

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    device='cpu'
    n_word_features = 2 # Number of features derived from every word in the input.
    window_size = 1
    n_features = (2 * window_size + 1) * n_word_features # Number of features used for every word in the input (including the window).
    max_length = 120 # longest sequence to parse
    n_classes = 5
    dropout = 0.5
    embed_size = 50
    hidden_size = 300
    batch_size = 32
    n_epochs = 15
    lr = 0.005

    def __init__(self, args):
        if "model_path" in args:
            # Where to save things.
            self.output_path = args.model_path
        else:
            self.output_path = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
        self.model_output = os.path.join(self.output_path, "model.weights")
        self.eval_output = os.path.join(self.output_path, "results.txt")
        self.conll_output = os.path.join(self.output_path, "predictions.conll")
        self.log_output = os.path.join(self.output_path, "log")
        self.device = int(args.device) if args.device != 'cpu' else args.device

class MultipleInputModel(torch.nn.Module):
    def __init__(self, helper, config, pretrained_embeddings):
        super(MultipleInputModel, self).__init__()
        self.config = config
        self.NN_model = config.NN_model
        self.linear = torch.nn.Linear(self.config.hidden_size, self.config.n_classes) # TODO: check new size for layer input

    def forward(self, sentences, meta_data_feat):
        NN_output = self.NN_model(sentences)
        input = torch.cat([NN_output, meta_data_feat], 1)
        # TODO: check axis of softmax
        # 4. Create softmax activations bc we're doing multi-class calssification
        out = F.softmax(self.linear(input))
        return out


class BiLstmModel(torch.nn.Module):
    """
    Implements a BiLSTM network with an embedding layer and
    single hidden layer.
    """
    def __init__(self, helper, config, pretrained_embeddings):
        super(BiLstmModel, self).__init__()
        self.config = config
        self._max_length = min(config.max_length, helper.max_length)

        self._dropout = torch.nn.Dropout(config.dropout)

        ### YOUR CODE HERE (3 lines)
        self.word_embeddings = torch.nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.lstm = torch.nn.LSTM(self.config.embed_size*self.config.n_features,
                                  self.config.hidden_size // 2,
                                  num_layers=1,
                                  bidirectional=True,
                                  batch_first=True)  # when input data is of shape (batch_size, seq_len, features)

        ### END YOUR CODE

    def forward(self, sentences):
        batch_size, seq_length = sentences.shape[0], sentences.shape[1]
        ### YOUR CODE HERE (5-9 lines)

        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, n_features) -> (batch_size, seq_len, embedding_dim)
        embeds = self.word_embeddings(sentences.long()) # dim: batch_size x batch_max_len x embedding_dim
        embeds = embeds.view(batch_size, seq_length, -1)
        drop = self._dropout(embeds)

        # 2. Run through RNN
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, lstm_hidden_dim)
        lstm_out, hidden = self.lstm(drop)
        lstm_out_drop = self._dropout(lstm_out)
        ### END YOUR CODE

        return lstm_out_drop