import re
from numpy import array
from collections import defaultdict, Counter

from categories_sol import *
# DO NOT USE IF YOU WANT TO LEARN SOMETHING

MIN_FREQ = 3
def invert_dict(d):
    res = {}
    for k, v in d.items():
        res[v] = k
    return res

def read_conll_pos_file(path):
    """
        Takes a path to a file and returns a list of word/tag pairs
    """
    sents = []
    with open(path, "r") as f:
        curr = []
        for line in f:
            line = line.strip()
            if line == "":
                sents.append(curr)
                curr = []
            else:
                tokens = line.strip().split("\t")
                curr.append((tokens[1],tokens[3]))
    return sents

def read_conll_ner_file(path):
    """
    Reads a path to a file @path in CoNLL file format.
    @returns a list of examples [(tokens), (labels)]. @tokens and @labels are lists of string.
    """
    sents = []
    with open(path, "r") as fstream:
        curr = []
        for line in fstream:
            line = line.strip()
            if len(line) == 0 or line.startswith("-DOCSTART-"):
                if len(curr) > 0:
                    sents.append(curr)
                    curr = []
            else:
                assert "\t" in line, r"Invalid CONLL format; expected a '\t' in {}".format(line)
                tok, lbl = line.split("\t")
                curr.append((tok, lbl))
    return sents

NER_LBLS = ["PER", "ORG", "LOC", "MISC", "O"]
def evaluate_ner(gold_tag_seqs, pred_tag_seqs):
    """Evaluates model performance on @examples.

    This function uses the model to predict labels for @examples and constructs a confusion matrix.

    Returns:
        The F1 score for predicting tokens as named entities.
    """
    token_cm = ConfusionMatrix(labels=NER_LBLS)

    correct_preds, total_correct, total_preds = 0., 0., 0.
    for gold_tags, pred_tags  in zip(gold_tag_seqs, pred_tag_seqs):
        for l, l_ in zip(gold_tags, pred_tags):
            token_cm.update(NER_LBLS.index(l), NER_LBLS.index(l_))
        gold = set(get_chunks(gold_tags))
        pred = set(get_chunks(pred_tags))
        correct_preds += len(gold.intersection(pred))
        total_preds += len(pred)
        total_correct += len(gold)

    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

    print("Token-level confusion matrix:\n" + token_cm.as_table())
    print("Token-level scores:\n" + token_cm.summary())
    print("Entity level P/R/F1: {:.2f}/{:.2f}/{:.2f}".format(*(p, r, f1)))
    return token_cm, (p, r, f1)

def to_table(data, row_labels, column_labels, precision=2, digits=4):
    """Pretty print tables.
    Assumes @data is a 2D array and uses @row_labels and @column_labels
    to display table.
    """
    # Convert data to strings
    data = [["%04.2f"%v for v in row] for row in data]
    cell_width = max(
        max(map(len, row_labels)),
        max(map(len, column_labels)),
        max(max(map(len, row)) for row in data))
    def c(s):
        """adjust cell output"""
        return s + " " * (cell_width - len(s))
    ret = ""
    ret += "\t".join(map(c, column_labels)) + "\n"
    for l, row in zip(row_labels, data):
        ret += "\t".join(map(c, [l] + row)) + "\n"
    return ret

class ConfusionMatrix(object):
    """
    A confusion matrix stores counts of (true, guessed) labels, used to
    compute several evaluation metrics like accuracy, precision, recall
    and F1.
    """

    def __init__(self, labels, default_label=None):
        self.labels = labels
        self.default_label = default_label if default_label is not None else len(labels) -1
        self.counts = defaultdict(Counter)

    def update(self, gold, guess):
        """Update counts"""
        self.counts[gold][guess] += 1

    def as_table(self):
        """Print tables"""
        # Header
        data = [[self.counts[l][l_] for l_,_ in enumerate(self.labels)] for l,_ in enumerate(self.labels)]
        return to_table(data, self.labels, ["go\\gu"] + self.labels)

    def summary(self, quiet=False):
        """Summarize counts"""
        keys = range(len(self.labels))
        data = []
        macro = array([0., 0., 0., 0.])
        micro = array([0., 0., 0., 0.])
        default = array([0., 0., 0., 0.])
        for l in keys:
            tp = self.counts[l][l]
            fp = sum(self.counts[l_][l] for l_ in keys if l_ != l)
            tn = sum(self.counts[l_][l__] for l_ in keys if l_ != l for l__ in keys if l__ != l)
            fn = sum(self.counts[l][l_] for l_ in keys if l_ != l)

            acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
            prec = (tp)/(tp + fp) if tp > 0  else 0
            rec = (tp)/(tp + fn) if tp > 0  else 0
            f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0

            # update micro/macro averages
            micro += array([tp, fp, tn, fn])
            macro += array([acc, prec, rec, f1])
            if l != self.default_label: # Count count for everything that is not the default label!
                default += array([tp, fp, tn, fn])

            data.append([acc, prec, rec, f1])

        # micro average
        tp, fp, tn, fn = micro
        acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
        prec = (tp)/(tp + fp) if tp > 0  else 0
        rec = (tp)/(tp + fn) if tp > 0  else 0
        f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0
        data.append([acc, prec, rec, f1])
        # Macro average
        data.append(macro / len(keys))

        # default average
        tp, fp, tn, fn = default
        acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
        prec = (tp)/(tp + fp) if tp > 0  else 0
        rec = (tp)/(tp + fn) if tp > 0  else 0
        f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0
        data.append([acc, prec, rec, f1])

        # Macro and micro average.
        return to_table(data, self.labels + ["micro","macro","not-O"], ["label", "acc", "prec", "rec", "f1"])

def get_chunks(seq, default=NER_LBLS.index("O")):
    """Breaks input of 4 4 4 0 0 4 0 ->   (0, 4, 5), (0, 6, 7)"""
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None
        # End of a chunk + start of a chunk!
        elif tok != default:
            if chunk_type is None:
                chunk_type, chunk_start = tok, i
            elif tok != chunk_type:
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
    return chunks

def increment_count(count_dict, key):
    """
        Puts the key in the dictionary if does not exist or adds one if it does.
        Args:
            count_dict: a dictionary mapping a string to an integer
            key: a string
    """
    if key in count_dict:
        count_dict[key] += 1
    else:
        count_dict[key] = 1

def compute_vocab_count(sents):
    """
        Takes a corpus and computes all words and the number of times they appear
    """
    vocab = {}
    for sent in sents:
        for token in sent:
            increment_count(vocab, token[0])
    return vocab

def replace_word(word):
    """
        Replaces rare words with categories (numbers, dates, etc...)
    """
    ### YOUR CODE HERE
    for cat in rare_words_transformation_rules:
        if cat[1](word):
            return cat[0]
    ### END YOUR CODE
    return "UNK"

def preprocess_sent(vocab, sents):
    """
        return a sentence, where every word that is not frequent enough is replaced
    """
    res = []
    total, replaced = 0, 0
    for sent in sents:
        new_sent = []
        for token in sent:
            if token[0] in vocab and vocab[token[0]] >= MIN_FREQ:
                new_sent.append(token)
            else:
                new_sent.append((replace_word(token[0]), token[1]))
                replaced += 1
            total += 1
        res.append(new_sent)
    print("replaced: " + str(float(replaced)/total))
    return res
