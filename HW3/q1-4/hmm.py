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
    e_word_tag_counts = Counter()
    for sent in sents:
        sent_e_word_tag = Counter(sent)
        e_word_tag_counts += sent_e_word_tag
        sent = [('*', '*'), ('*', '*')] + sent + [('END', 'END')]
        for i in range(2, len(sent)):
            #  e_word_tag_counts[sent[i]] += 1
            if i < len(sent)-1:
                e_tag_counts[sent[i][1]] += 1
            q_tri_counts[(sent[i - 2][1], sent[i - 1][1], sent[i][1])] += 1
            q_bi_counts[(sent[i - 1][1], sent[i][1])] += 1
            q_uni_counts[sent[i][1]] += 1
            total_tokens += 1
        q_uni_counts['*'] += 2
        q_bi_counts[('*', '*')] += 1

    ### END YOUR CODE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts


def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                e_word_tag_counts, e_tag_counts, lambda1, lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    n = len(sent)
    predicted_tags = ["O"] * (n)
    ### YOUR CODE HERE
    pai = defaultdict(dict)
    pai[-1][("*", "*")] = 1
    S = e_tag_counts.keys()
    bp = {}
    for k in range(n): # word index in sentence

        if k==0:
            S_w, S_u, S_v = ['*'], ['*'], S
        elif k==1:
            S_w, S_u, S_v = ['*'], S, S
        else:
            S_w, S_u, S_v = S, S, S

        for u in S_u:  # previous tag
            for v in S_v:  # current tag
                max_val = -float('Inf')
                best_tag = ""
                for w in S_w:  # previous, previous tag
                    q = lambda1 * (q_tri_counts.get((w, u, v), 0) / q_bi_counts.get((u, v), float('Inf'))) + \
                       lambda2 * (q_bi_counts.get((u, v), 0) / q_uni_counts.get(v, float('Inf'))) + \
                       (1 - lambda1 - lambda2) * (q_uni_counts.get(v, 0) / total_tokens)
                    e = e_word_tag_counts.get((sent[k][0], v), 0) /e_tag_counts.get(v, float('Inf'))
                    pai_w = pai[k-1][(w, u)] * q * e
                    if pai_w > max_val:
                        max_val = pai_w
                        best_tag = w
                pai[k][(u, v)] = max_val
                bp[(k, u, v)] = best_tag

    max_bp_val = -float('Inf')

    for u_, v_ in pai[n-1]:
        q = lambda1 * (q_tri_counts.get((u_, v_, 'END'), 0) / q_bi_counts.get((v_, 'END'), float('Inf'))) + \
            lambda2 * (q_bi_counts.get((v_, 'END'), 0) / q_uni_counts.get('END', float('Inf'))) + \
            (1 - lambda1 - lambda2) * (q_uni_counts.get('END', 0) / total_tokens)
        p = pai[n - 1][(u_, v_)]*q
        if p > max_bp_val:
            u, v = u_, v_
            max_bp_val = p

    if n==1:
        predicted_tags[n-1] = v
    else:
        predicted_tags[-1], predicted_tags[-2] = v, u
        for k in range(n-3, -1, -1):
            predicted_tags[k] = bp[(k+2, predicted_tags[k+1], predicted_tags[k+2])]

    ### END YOUR CODE
    return predicted_tags

def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    import numpy as np
    hyper_search = {}
    for lmbd1 in np.arange(0, 1.05, 0.05):
        for lmbd2 in np.arange(0, 1.05, 0.05):
            if lmbd1+lmbd2<=1:
                print("Start evaluation")
                gold_tag_seqs = []
                pred_tag_seqs = []
                for sent in test_data:
                    words, true_tags = zip(*sent)
                    gold_tag_seqs.append(true_tags)

                    ### YOUR CODE HERE
                    prediction_list = hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                                e_word_tag_counts, e_tag_counts, lambda1=lmbd1, lambda2=lmbd2)
                    pred_tag_seqs.append(prediction_list)
                    ### END YOUR CODE
            token_cm, scores = evaluate_ner(gold_tag_seqs, pred_tag_seqs)
            hyper_search[(lmbd1, lmbd2)] = scores[0]
            with open('hyper_search_lambda.txt', 'a') as f:
                f.write("{},{},{}".format(lmbd1, lmbd2, scores[0]))
    print(max(hyper_search, key=hyper_search.get))

    #return evaluate_ner(gold_tag_seqs, pred_tag_seqs)


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