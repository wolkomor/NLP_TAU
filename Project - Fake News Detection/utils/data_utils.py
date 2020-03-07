import torch
import os
import re
import random
from torchtext import data
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from torchtext.vocab import Vectors, GloVe
from gensim.models import Doc2Vec
import multiprocessing

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    text = "<soa> " + text + " <eoa>"
    # Clean the text
    re_url = re.compile(r"(www|http[s]?://)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                        re.MULTILINE | re.UNICODE)
    re_ip = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
    re_date = re.compile(r".+\d{1,4}(\/|.|\-|\\)\d{1,2}(\/|\.|\-|\\)\d{1,4}")
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=<>]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"<br \/>", " ", text) #replace '<br \/>'with single space
    text = re.sub(r"\'s+", " ", text) #replace multiple spaces with single space
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"\!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"\;", " ", text)
    text = re.sub(r"\:", " ", text)
    text = re.sub(r"\(", " ", text)
    text = re.sub(r"\)", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re_url.sub("URL", text)
    text = re_ip.sub("IPADDRESS", text)
    text = re_date.sub(" DATE", text)
    text = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", "NUMBER", text)

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words + shorten words to their stems
    stops = set(stopwords.words("english"))
    stemmer = SnowballStemmer('english')
    if remove_stopwords and stem_words:
        text = [stemmer.stem(w) for w in text if not w in stops]
    elif remove_stopwords and not stem_words:
        text = [w for w in text if not w in stops]
    elif not remove_stopwords and stem_words:
        text = [stemmer.stem(w) for w in text]

    #text = " ".join(text)

    # Return a list of words
    return text


def load_dataset(path, glove_dim, doc_length, SEED):
    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fix_length which
                 will pad each sequence to have a fix length of doc_length.

    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.

    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.

    """

    # loading custom dataset
    #initialize glove embeddings
    #tokenize = lambda x: x.split()
    #tokenize = get_tokenizer("basic_english")
    TEXT = data.Field(sequential=True, tokenize=text_to_wordlist, lower=False, include_lengths=True, batch_first=True,
                      fix_length=None, init_token="<SOA>", eos_token="<EOA>")
    LABEL = data.LabelField()
    #fields = [('text', TEXT) ('class', LABEL)]
    #train_data = data.Dataset(examples=,fields=fields)

    fields = [('text', TEXT), ('label', LABEL)]
    train_data = data.TabularDataset(path=path, format='csv', fields=fields, skip_header=True)

    # Initialized embedding
    if not os.path.exists(".vector_cache"):
        os.mkdir(".vector_cache")

    train_data, valid_data = train_data.split(split_ratio=0.7, random_state=random.seed(SEED))

    cores = multiprocessing.cpu_count()
    PV_DBOW_model = Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=2, sample=0,
            epochs=20, workers=cores)
    PV_DBOW_model.build_vocab(train_data)

    TEXT.build_vocab(train_data, min_freq=3, vectors=GloVe(name='6B', dim=glove_dim))
                     #vectors = Vectors(name=f"\data\glove.6B\glove.6B.{glove_dim}d.txt"))
    LABEL.build_vocab(train_data)

    #printing
    word_embeddings = TEXT.vocab.vectors #pretrained_embedding
    print("Length of Text Vocabulary: " + str(len(TEXT.vocab))) #No. of unique tokens in text
    print("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print("Label Length: " + str(len(LABEL.vocab))) #No. of unique tokens in label
    print(TEXT.vocab.freqs.most_common(10)) #Commonly used words
    print(TEXT.vocab.stoi) #Word dictionary

    # train_data, valid_data = train_data.split(split_ratio=0.7, random_state=random.seed(SEED))  # Further splitting of training_data to create new training_data & validation_data
    train_iter, valid_iter = data.BucketIterator.splits((train_data, valid_data), batch_size=32,
                                                                   sort_key=lambda x: len(x.text), repeat=False,
                                                                   shuffle=True)

    vocab_size = len(TEXT.vocab)
    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter