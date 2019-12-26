import os
from data import *
from collections import defaultdict

def most_frequent_train(train_data):
    """
    Gets training data that includes tagged sentences.
    Returns a dictionary that maps every word in the training set to its most frequent tag.
    The dictionary should have a default value.
    """
    ### YOUR CODE HERE
    counter = defaultdict(lambda: defaultdict(int))
    most_freq_tag = {}
    list_words = []
    for sent in train_data:
        for word, tag in sent:
            counter[word][tag] += 1
            list_words.append(word)
    print (len(set(list_words)))
    for word in counter:
        most_freq_tag[word] = max(counter[word], key=counter[word].get)
    return most_freq_tag
    ### END YOUR CODE

def most_frequent_eval(test_set, pred_tags):
    """
    Gets test data and tag prediction map.
    Returns an evaluation of the accuracy of the most frequent tagger.
    """
    gold_tag_seqs = []
    pred_tag_seqs = []
    for sent in test_set:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)
        pred_sent_tag = []
        ### YOUR CODE HERE
        for word in words:
            pred_sent_tag.append(pred_tags.get(word,pred_tags['UNK']))
        pred_tag_seqs.append(pred_sent_tag)
        ### END YOUR CODE

    return evaluate_ner(gold_tag_seqs, pred_tag_seqs)

if __name__ == "__main__":
    train_sents = read_conll_ner_file(r"data/train.conll")
    dev_sents = read_conll_ner_file(r"data/dev.conll")
    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    model = most_frequent_train(train_sents)
    most_frequent_eval(dev_sents, model)

