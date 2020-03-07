import torch
import torch.nn.functional as F
from torch import nn

def from_pretrained(self, embeddings, freeze=True):

    # self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_length)  # Initializing the look-up table.
    # self.word_embeddings.weight = torch.nn.Parameter(embeddings, requires_grad=False)  # Assigning the look-up table to the pre-trained GloVe word embedding.

    assert embeddings.dim() == 2, \
         'Embeddings parameter is expected to be 2-dimensional'
    vocab_size, emb_dim = embeddings.shape
    embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
    embedding.weight = torch.nn.Parameter(embeddings)
    embedding.weight.requires_grad = not freeze
    return embedding

class BiLstmModel(torch.nn.Module):
    """
    Implements a BiLSTM network with an embedding layer and
    single hidden layer.
    """
    def __init__(self, hidden_size, config, pretrained_embeddings):
        super(BiLstmModel, self).__init__()
        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        hidden_size : Size of the hidden_state of the LSTM
        # vocab_size : Size of the vocabulary containing unique words
        # embed_size : Embeddding dimension of GloVe word embeddings
        # pretrained_embeddings : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
        """
        self.n_classes = config.n_classes
        self._dropout = torch.nn.Dropout(config.dropout)
        #vocab_size, emb_dim = pretrained_embeddings.shape
        #self.word_embeddings = torch.nn.Embedding(vocab_size, emb_dim)
        #self.word_embeddings.weight = nn.Parameter(pretrained_embeddings, requires_grad=False)  # Assigning the look-up table to the pre-trained GloVe word embedding.
        self.word_embeddings = torch.nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.lstm = torch.nn.LSTM(config.embedding_dim,
                                  hidden_size // 2,
                                  num_layers=1,
                                  bidirectional=True,
                                  batch_first=True)  # when input data is of shape (batch_size, seq_len, features)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = torch.nn.Linear(hidden_size, config.n_classes)

    def forward(self, articles,text_lengths):
        batch_size, seq_length = articles.shape[0], articles.shape[1] # dim: batch_size x batch_max_len

        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, n_features) -> (batch_size, seq_len, embedding_dim)
        embeds = self.word_embeddings(articles.long()) # dim: batch_size x batch_max_len x embedding_dim
        drop = self._dropout(embeds)

        # packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(drop, text_lengths, batch_first=True)

        # 2. Run through RNN
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, lstm_hidden_dim)
        lstm_out, hidden = self.lstm(packed_embedded)
        lstm_out, hidden = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True) # dim: batch_size x batch_max_len x embedding_dim
        lstm_out_drop = self._dropout(lstm_out) # dim: batch_size x batch_max_len x embedding_dim

        # 3. Project to tag space
        # we need to reshape the data so it goes into the linear layer
        # Dim: (batch_size * seq_len, lstm_hidden_dim)
        lstm_out_drop = lstm_out_drop.contiguous()
        lstm_out_drop = lstm_out_drop.view(-1, lstm_out_drop.shape[2])

        tag_space = self.hidden2tag(lstm_out_drop)  # dim: batch_size * batch_max_len x num_tags

        # 4. Create softmax activations bc we're doing multi-class classification
        tag_probs = F.softmax(tag_space, dim=1)
        # reshape so we're back to (batch_size, seq_len, n_classes)
        tag_probs = tag_probs.view(batch_size, seq_length, self.n_classes)

        return tag_probs