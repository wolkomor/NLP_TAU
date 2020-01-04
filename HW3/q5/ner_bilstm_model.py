# -*- coding: utf-8 -*-

import argparse
import sys
import os
import time
import logging
from datetime import datetime

import torch

from util import print_sentence, read_conll, write_conll
from data_util import load_data, load_embeddings, ModelHelper
from defs import LBLS
from extras import TrainerBase, Predictor, Evaluator, BaseDataPreprocessor, AttrDict

logger = logging.getLogger("hw3.q5")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

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


class NerBiLstmModel(torch.nn.Module):
    """
    Implements a BiLSTM network with an embedding layer and
    single hidden layer.
    This network will predict a sequence of labels (e.g. PER) for a
    given token (e.g. Henry) using a featurized window around the token.
    """
    def __init__(self, helper, config, pretrained_embeddings):
        """
        TODO:
        - Initialize the layer of the models:
            - *Unfrozen* embeddings of shape (V, D) which are loaded with pre-trained weights (`pretrained_embeddings`)
            - BiLSTM layer with hidden size of H/2 per direction
            - Linear layer with output of shape C

            Where:
            V - size of the vocabulary
            D - size of a word embedding
            H - size of the hidden layer
            C - number of classes being predicted

        Hints:
        - For the input dimension of the BiLSTM, think about the size of an embedded word representation
        """
        super(NerBiLstmModel, self).__init__()
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

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = torch.nn.Linear(self.config.hidden_size, self.config.n_classes)

        ### END YOUR CODE

    def forward(self, sentences):
        """
        TODO:
        - Perform the forward pass of the model, according to the model description in the handout:
            1. Get the embeddings of the input
            2. Apply dropout on the output of 1
            3. Pass the output of 2 through the BiLSTM layer
            4. Apply dropout on the output of 3
            5. Pass the output of 4 through the linear layer
            6. Perform softmax on the output of 5 to get tag_probs

        Hints:
        - Reshape the output of the embeddings layer so the full representation of an embedded word fits in one dimension.
          You might find the .view method of a tensor helpful.

        Args:
        - sentences: The input tensor of shape (batch_size, max_length, n_features)

        Returns:
        - tag_probs: A tensor of shape (batch_size, max_length, n_classes) which represents the probability
                     for each tag for each word in a sentence.
        """

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

        # 3. Project to tag space
        # we need to reshape the data so it goes into the linear layer
        # Dim: (batch_size * seq_len, lstm_hidden_dim)
        lstm_out_drop = lstm_out_drop.contiguous()
        lstm_out_drop = lstm_out_drop.view(-1, lstm_out_drop.shape[2])

        tag_space = self.hidden2tag(lstm_out_drop) # dim: batch_size * batch_max_len x num_tags

        # 4. Create logsoftmax activations bc we're doing multi-class calssification
        tag_probs = torch.nn.functional.log_softmax(tag_space, dim=1)  # dim: batch_size * seq_len x n_classes

        # reshape so we're back to (batch_size, seq_len, n_classes)
        tag_probs = tag_probs.view(batch_size, seq_length, self.config.n_classes)

        ### END YOUR CODE
        return tag_probs

class Trainer(TrainerBase):
    def __init__(self, model, config, helper, logger):
        """
        TODO:
        - Define the cross entropy loss function in self._loss_function.
          It will be used in _batch_loss.

        Hints:
        - Don't use automatically PyTorch's CrossEntropyLoss - read its documentation first 
        """
        super(Trainer, self).__init__(model, config, helper, logger)

        ### YOUR CODE HERE (1 line)
        self._loss_function = torch.nn.NLLLoss()
        ### END YOUR CODE
        self._optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    def _train_on_batch(self, sentences, labels, masks):
        model = self._model
        config = self._config

        tag_probs = model(sentences)

        model.zero_grad()
        batch_loss = self._batch_loss(tag_probs, labels, masks)
        batch_loss.backward()

        self._optimizer.step()

        return batch_loss

    def _batch_loss(self, tag_probs, labels, masks):
        """
        TODO:
        - Calculate the cross entropy loss of the input batch

        Hints:
        - You might find torch.unsqueeze, torch.masked_fill (use ~masks to get the inversion) and torch.transpose useful.

        Args:
        - tag_probs: A tensor of shape (batch_size, max_length, n_classes) containing the output of the neural
                     network (*after* softmax).
        - labels: The gold labels tensor of shape (batch_size, max_length)
        - masks: The masks tensor of shape (batch_size, max_length)

        Returns:
            loss: A 0-d tensor (scalar)
        """
        ### YOUR CODE HERE (3-6 lines)
        masked_tag_probs = tag_probs[masks]
        masked_labels = labels[masks]
        masked_labels = masked_labels.long()
        ### END YOUR CODE
        loss = self._loss_function(masked_tag_probs, masked_labels)
        return loss

class DataPreprocessor(BaseDataPreprocessor):
    def pad_sequences(self, examples):
        """Ensures each input-output seqeunce pair in @data is of length
        @max_length by padding it with zeros and truncating the rest of the
        sequence.

        TODO: In the code below, for every sentence, labels pair in @data,
        (a) create a new sentence which appends zero feature vectors until
        the sentence is of length @max_length. If the sentence is longer
        than @max_length, simply truncate the sentence to be @max_length
        long.
        (b) create a new label sequence similarly.
        (c) create a _masking_ sequence that has a True wherever there was a
        token in the original sequence, and a False for every padded input.

        Example: for the (sentence, labels) pair: [[4,1], [6,0], [7,0]], [1,
        0, 0], and max_length = 5, we would construct
            - a new sentence: [[4,1], [6,0], [7,0], [0,0], [0,0]]
            - a new label seqeunce: [1, 0, 0, 4, 4], and
            - a masking seqeunce: [True, True, True, False, False].

        Args:
            data: is a list of (sentence, labels) tuples. @sentence is a list
                containing the words in the sentence and @label is a list of
                output labels. Each word is itself a list of
                @n_features features. For example, the sentence "Chris
                Manning is amazing" and labels "PER PER O O" would become
                ([[1,9], [2,9], [3,8], [4,8]], [1, 1, 4, 4]). Here "Chris"
                the word has been featurized as "[1, 9]", and "[1, 1, 4, 4]"
                is the list of labels. 
            max_length: the desired length for all input/output sequences.
        Returns:
            a new list of data points of the structure (sentence, labels, mask).
            Each of sentence, labels and mask are of length @max_length.
            See the example above for more details.
        """
        ret = []

        max_length = self._max_length
        # Use this zero vector when padding sequences.
        zero_vector = [0] * self._n_features
        zero_label = 4 # corresponds to the 'O' tag

        for sentence, labels in examples:
            ### YOUR CODE HERE (~5 lines)
            n = len(sentence)
            if len(sentence)>max_length:
                ret.append([sentence[:max_length], labels[:max_length], [True] * max_length])
            else:
                ret.append([sentence+[zero_vector]*(max_length-n), labels+[zero_label]*(max_length-n), [True] *n + [False]*(max_length-n)])
            ### END YOUR CODE
        return ret

def do_training(args):
    torch.manual_seed(133)
    # Set up configuration and output
    config = Config(args)
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)

    # Set up logging
    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    # Load data
    helper, data = load_data(args)
    train_examples = data['train_examples']
    dev_examples = data['dev_examples']
    helper.save(config.output_path)

    # Load embeddings
    embeddings = load_embeddings(args, helper, config.device)

    # Initialize model
    logger.info("Initializing model...",)
    model = NerBiLstmModel(helper, config, embeddings)
    model.to(config.device)

    # Preprocess data
    data_preprocessor = DataPreprocessor(model, config, helper)
    train_examples = data_preprocessor.preprocess_sequence_data(train_examples)
    dev_examples = data_preprocessor.preprocess_sequence_data(dev_examples)

    # Start training
    trainer = Trainer(model, config, helper, logger)
    logger.info("Starting training...",)
    trainer.train(train_examples, dev_examples)

    # Save predictions of the best model
    logger.info("Training completed, saving predictions of the best model...",)
    with torch.no_grad():
        model.load_state_dict(torch.load(config.model_output))
        model.eval()
        predictor = Predictor(model, config)
        output = predictor.predict(dev_examples, use_str_labels=True)
        sentences, labels, predictions = zip(*output)
        predictions = [[LBLS[l] for l in preds] for preds in predictions]
        output = list(zip(sentences, labels, predictions))

        with open(model.config.conll_output, 'w') as f:
            write_conll(f, output)
        with open(model.config.eval_output, 'w') as f:
            for sentence, labels, predictions in output:
                print_sentence(f, sentence, labels, predictions)

def do_evaluate(args):
    config = Config(args)
    helper = ModelHelper.load(args.model_path)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    # Initialize model
    model = NerBiLstmModel(helper, config, embeddings)
    model.to(config.device)

    # Load data
    helper, data = load_data(args, helper)
    examples = data['examples']

    # Preprocess data
    data_preprocessor = DataPreprocessor(model, config, helper)
    examples = data_preprocessor.preprocess_sequence_data(examples)

    with torch.no_grad():
        model.load_state_dict(torch.load(config.model_output))
        model.eval()

        evaluator = Evaluator(Predictor(model, config))
        token_cm, entity_scores = evaluator.evaluate(examples)
        print("Token-level confusion matrix:\n" + token_cm.as_table())
        print("Token-level scores:\n" + token_cm.summary())
        print("Entity level P/R/F1: {:.2f}/{:.2f}/{:.2f}".format(*entity_scores))

def do_predict(args):
    config = Config(args)
    helper = ModelHelper.load(args.model_path)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    # Initialize model
    model = NerBiLstmModel(helper, config, embeddings)
    model.to(config.device)

    # Load data
    helper, data = load_data(args, helper)
    examples = data['examples']

    # Preprocess data
    data_preprocessor = DataPreprocessor(model, config, helper)
    examples = data_preprocessor.preprocess_sequence_data(examples)

    with torch.no_grad():
        model.load_state_dict(torch.load(config.model_output))
        model.eval()

        predictor = Predictor(model, config)
        output = predictor.predict(examples, use_str_labels=True)
        sentences, labels, predictions = zip(*output)
        predictions = [[LBLS[l] for l in preds] for preds in predictions]
        output = list(zip(sentences, labels, predictions))

        for sentence, labels, predictions in output:
            print_sentence(args.output, sentence, labels, predictions)

def do_padding_test(_):
    logger.info("Testing pad_sequences")
    model, config = AttrDict({'_max_length': 4}), AttrDict({'n_features': 2, 'window_size': 1})
    data_preprocessor= DataPreprocessor(model, config, None)
    data = [
        ([[4,1], [6,0], [7,0]], [1, 0, 0]),
        ([[3,0], [3,4], [4,5], [5,3], [3,4]], [0, 1, 0, 2, 3]),
        ]
    ret = [
        ([[4,1], [6,0], [7,0], [0,0]], [1, 0, 0, 4], [True, True, True, False]),
        ([[3,0], [3,4], [4,5], [5,3]], [0, 1, 0, 2], [True, True, True, True])
        ]

    ret_ = data_preprocessor.pad_sequences(data)
    assert len(ret_) == 2, "Did not process all examples: expected {} results, but got {}.".format(2, len(ret_))
    for i in range(2):
        assert len(ret_[i]) == 3, "Did not populate return values corrected: expected {} items, but got {}.".format(3, len(ret_[i]))
        for j in range(3):
            assert ret_[i][j] == ret[i][j], "Expected {}, but got {} for {}-th entry of {}-th example".format(ret[i][j], ret_[i][j], j, i)
    logger.info("Passed!")

def do_training_test(args):
    logger.info("Testing implementation of NerBiLstmModel")
    torch.manual_seed(133)
    # Set up configuration and output
    config = Config(args)
    config.n_epochs = 1
    config.model_output = None

    # Load data
    helper, data = load_data(args)
    train_examples = data['train_examples']
    dev_examples = data['dev_examples']

    # Load embeddings
    embeddings = load_embeddings(args, helper, config.device)

    # Initialize model
    model = NerBiLstmModel(helper, config, embeddings)
    model.to(config.device)

    # Preprocess data
    data_preprocessor = DataPreprocessor(model, config, helper)
    train_examples = data_preprocessor.preprocess_sequence_data(train_examples)
    dev_examples = data_preprocessor.preprocess_sequence_data(dev_examples)

    # Start training
    trainer = Trainer(model, config, helper, logger)
    logger.info("Starting training...",)
    trainer.train(train_examples, dev_examples)

    logger.info("Model did not crash!")
    logger.info("Passed!")

def main(arguments_str):  
    args = sys.argv[1:]
    if arguments_str:
        args = arguments_str.split()
    
    parser = argparse.ArgumentParser(description='Trains and tests an NER model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('test_padding', help='')
    command_parser.set_defaults(func=do_padding_test)

    command_parser = subparsers.add_parser('test_training', help='')
    command_parser.add_argument('-dt', '--data-train', type=argparse.FileType('r'), default="data/tiny.conll", help="Training data")
    command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), default="data/tiny.conll", help="Dev data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument('--device', type=str, default="cpu", help="Device to use")
    command_parser.set_defaults(func=do_training_test)

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-dt', '--data-train', type=argparse.FileType('r'), default="data/train.conll", help="Training data")
    command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), default="data/dev.conll", help="Dev data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument('--device', type=str, default="cpu", help="Device to use")
    command_parser.set_defaults(func=do_training)

    command_parser = subparsers.add_parser('evaluate', help='')
    command_parser.add_argument('-d', '--data', type=argparse.FileType('r'), default="data/dev.conll", help="Data")
    command_parser.add_argument('-m', '--model-path', help="Model path")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument('--device', type=str, default="cpu", help="Device to use")
    command_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout, help="Output file")
    command_parser.set_defaults(func=do_evaluate)

    command_parser = subparsers.add_parser('predict', help='')
    command_parser.add_argument('-d', '--data', type=argparse.FileType('r'), default="data/dev.conll", help="Data")
    command_parser.add_argument('-m', '--model-path', help="Model path")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument('--device', type=str, default="cpu", help="Device to use")
    command_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout, help="Output file")
    command_parser.set_defaults(func=do_predict)

    ARGS = parser.parse_args(args)
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)

if __name__ == "__main__":
    main('train')

