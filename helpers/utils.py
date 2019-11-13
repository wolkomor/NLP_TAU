#!/usr/bin/env python

import numpy as np


def normalize_rows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    N = x.shape[0]
    x /= np.sqrt(np.sum(x**2, axis=1)).reshape((N,1)) + 1e-30
    return x


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s


def get_negative_samples(outside_word_idx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    neg_sample_word_indices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outside_word_idx:
            newidx = dataset.sampleTokenIdx()
        neg_sample_word_indices[k] = newidx
    return neg_sample_word_indices
