import re

import numpy as np


def invert_dict(d):
    return {v: k for k, v in d.items()}


def docs_to_indices(docs, word_to_num):
    docs = [_pad_sequence(seq, left=2, right=1) for seq in docs]
    ret = []
    for seq in docs:
        words = [_canonicalize_word(wt[0], word_to_num) for wt in seq]
        ret.append(_seq_to_indices(words, word_to_num))

    # return as numpy array for fancier slicing
    return np.array(ret, dtype=object)


def load_dataset(fname):
    docs = []
    with open(fname) as fd:
        cur = []
        for line in fd:
            # new sentence on -DOCSTART- or blank line
            if re.match(r"-DOCSTART-.+", line) or (len(line.strip()) == 0):
                if len(cur) > 0:
                    docs.append(cur)
                cur = []
            else:  # read in tokens
                cur.append(line.strip().split("\t",1))
        # flush running buffer
        docs.append(cur)
    return docs


def _canonicalize_digits(word):
    if any([c.isalpha() for c in word]):
        return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "")  # remove thousands separator
    return word


def _canonicalize_word(word, wordset=None, digits=True):
    word = word.lower()
    if digits:
        if (wordset is not None) and (word in wordset):
            return word
        word = _canonicalize_digits(word)  # try to canonicalize numbers
    if (wordset is None) or (word in wordset):
        return word
    else:
        return "UUUNKKK"  # unknown token


def _pad_sequence(seq, left=1, right=1):
    return left*[("<s>", "")] + seq + right*[("</s>", "")]


def _seq_to_indices(words, word_to_num):
    return np.array([word_to_num[w] for w in words])



