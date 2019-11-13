#!/usr/bin/env python

import sys
import time

import matplotlib
import matplotlib.pyplot as plt

from helpers.treebank import StanfordSentiment
from q3c_word2vec import *
from q3d_sgd import *

# Check Python Version
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 5

matplotlib.use('agg')

# Reset the random seed to make sure that everyone gets the same results
random.seed(1987)
dataset = StanfordSentiment()
tokens = dataset.tokens()
n_words = len(tokens)

# We are going to train 10-dimensional vectors for this assignment
dim_vectors = 10

# Context size
C = 5

# Reset the random seed to make sure that everyone gets the same results
random.seed(1407)
np.random.seed(1411)

start_time = time.time()
word_vectors = np.concatenate(
    ((np.random.rand(n_words, dim_vectors) - 0.5) /
     dim_vectors, np.zeros((n_words, dim_vectors))),
    axis=0)
word_vectors = sgd(
    lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C, neg_sampling_loss_and_gradient),
    word_vectors, 0.3, 40000, None, True, print_every=10)
# Note that normalization is not called here. This is not a bug,
# normalizing during training loses the notion of length.

print("sanity check: cost at convergence should be around or below 10")
print("training took {} seconds".format(time.time() - start_time))

# concatenate the input and output word vectors
word_vectors = np.concatenate(
    (word_vectors[:n_words, :], word_vectors[n_words:, :]),
    axis=1)

visualize_words = [
    "great", "cool", "brilliant", "wonderful", "well", "amazing", "worth", "sweet", "enjoyable",
    "boring", "bad", "dumb", "annoying",
    "female", "male", "queen", "king", "man", "woman",
    "rain", "snow", "hail",
    "coffee", "tea"]

visualize_idx = [tokens[word] for word in visualize_words]
visualize_vecs = word_vectors[visualize_idx, :]
temp = (visualize_vecs - np.mean(visualize_vecs, axis=0))
covariance = 1.0 / len(visualize_idx) * temp.T.dot(temp)
U, S, V = np.linalg.svd(covariance)
coord = temp.dot(U[:, 0:2])

for i in range(len(visualize_words)):
    plt.text(coord[i, 0], coord[i, 1], visualize_words[i],
             bbox=dict(facecolor='green', alpha=0.1))

plt.xlim((np.min(coord[:, 0]), np.max(coord[:, 0])))
plt.ylim((np.min(coord[:, 1]), np.max(coord[:, 1])))

plt.savefig('word_vectors.png')
