import torch
import torch.nn.functional as F

class MultipleInputModel(torch.nn.Module):
    def __init__(self, config):
        super(MultipleInputModel, self).__init__()
        self.NN_model = config.NN_model
        self.linear = torch.nn.Linear(config.hidden_size, config.n_classes) # TODO: check new size for layer input

    def forward(self, sentences, meta_data_feat):
        NN_output = self.NN_model(sentences)
        input = torch.cat([NN_output, meta_data_feat], 1)
        # TODO: check axis of softmax
        # 4. Create softmax activations bc we're doing binary calssification
        tag_probs = F.softmax(self.linear(input))
        return tag_probs

class BiLstmModel(torch.nn.Module):
    """
    Implements a BiLSTM network with an embedding layer and
    single hidden layer.
    """
    def __init__(self, config, pretrained_embeddings):
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
        self._dropout = torch.nn.Dropout(config.dropout)
        self.word_embeddings = torch.nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.lstm = torch.nn.LSTM(config.embedding_dim,
                                  config.hidden_size // 2,
                                  num_layers=1,
                                  bidirectional=True,
                                  batch_first=True)  # when input data is of shape (batch_size, seq_len, features)

    def forward(self, articles):
        batch_size, seq_length = articles.shape[0], articles.shape[1]

        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, n_features) -> (batch_size, seq_len, embedding_dim)
        embeds = self.word_embeddings(articles.long()) # dim: batch_size x batch_max_len x embedding_dim
        embeds = embeds.view(batch_size, seq_length, -1)
        drop = self._dropout(embeds)

        # 2. Run through RNN
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, lstm_hidden_dim)
        lstm_out, hidden = self.lstm(drop)
        lstm_out_drop = self._dropout(lstm_out)

        return lstm_out_drop