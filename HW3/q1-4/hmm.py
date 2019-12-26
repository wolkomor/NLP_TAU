import os
import time
from data import *
from collections import defaultdict, Counter

def hmm_train(sents):
    """
        sents: list of tagged sentences
        Returns: the q-counts and e-counts of the sentences' tags, total number of tokens in the sentences
    """

    print("Start training")
    total_tokens = 0
    # YOU MAY OVERWRITE THE TYPES FOR THE VARIABLES BELOW IN ANY WAY YOU SEE FIT
    q_tri_counts, q_bi_counts, q_uni_counts, e_tag_counts = [defaultdict(int) for i in range(4)]
    e_word_tag_counts = defaultdict(lambda: defaultdict(int))
    ### YOUR CODE HERE
    for sent in sents:
        sent = [('*', '*'), ('*', '*')] + sent + [('END', 'END')]
        for i in range(2,len(sent)):
            e_word_tag_counts[sent[i]] += 1
            e_tag_counts[sent[i][1]] += 1
            q_tri_counts[(sent[i - 2][1], sent[i - 1][1], sent[i][1])] += 1
            q_bi_counts[(sent[i - 1][1], sent[i][1])] += 1
            q_uni_counts[sent[i][1]] += 1
            total_tokens += 1

    ### END YOUR CODE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts


def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                e_word_tag_counts, e_tag_counts, lambda1, lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE
    sent = [('*', '*'), ('*', '*')] + sent + [('END', 'END')]
    pai = {}
    S = e_tag_counts.keys()
    for i in range(2, len(sent)):
        for u in S:
            for v in S:
                for w in S:
                    q = lambda1 * (q_tri_counts[(w, u, v)] / q_bi_counts[(u, v)]) + \
                       lambda2 * (q_bi_counts[(u, v)] / q_uni_counts[v]) + \
                       (1 - lambda1 - lambda2) * (q_uni_counts[v] / total_tokens)
                    e = e_word_tag_counts[(sent[i], v)]/e_tag_counts[v]
                    pai[i, u, v] = pai[(i-1, w, u)] * q * e

    ### END YOUR CODE
    return predicted_tags

def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    print("Start evaluation")
    gold_tag_seqs = []
    pred_tag_seqs = []
    for sent in test_data:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        ### YOUR CODE HERE
        raise NotImplementedError
        ### END YOUR CODE

    return evaluate_ner(gold_tag_seqs, pred_tag_seqs)

if __name__ == "__main__":
    start_time = time.time()
    train_sents = read_conll_ner_file("data/train.conll")
    dev_sents = read_conll_ner_file("data/dev.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)

    hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
             e_word_tag_counts, e_tag_counts)

    train_dev_end_time = time.time()
    print("Train and dev evaluation elapsed: " + str(train_dev_end_time - start_time) + " seconds")
