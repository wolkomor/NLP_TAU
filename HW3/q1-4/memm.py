from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time
import os
import numpy as np
from collections import defaultdict

def build_extra_decoding_arguments(train_sents):
    """
    Receives: all sentences from training set
    Returns: all extra arguments which your decoding procedures requires
    """

    extra_decoding_arguments = defaultdict(set)
    ### YOUR CODE HERE
    for sent in train_sents:
        for word, tag in sent:
            extra_decoding_arguments[word].add(tag)
    ### END YOUR CODE

    return extra_decoding_arguments


def extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Returns: The word's features.
    """
    features = {}
    features['word'] = curr_word
    ### YOUR CODE HERE
    for i in range(min(5, len(curr_word))):
        features['prefixes_{}'.format(i+1)] = curr_word[:i+1]
        features['suffixes_{}'.format(i + 1)] = curr_word[-(i+1):]
    features['trigrams'] = prev_tag +"_" +prevprev_tag
    features['bigrams'] = prev_tag
    features['previous_word'] = prev_word
    features['previous_previous_word'] = prevprev_word
    features['subsequent_word'] = next_word
    ### END YOUR CODE
    return features

def extract_features(sentence, i):
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<st>', '*')
    prevprev_token = sentence[i - 2] if i > 1 else ('<st>', '*')
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', 'STOP')
    return extract_features_base(curr_word, next_token[0], prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[1])

def vectorize_features(vec, features):
    """
        Receives: feature dictionary
        Returns: feature vector

        Note: use this function only if you chose to use the sklearn solver!
        This function prepares the feature vector for the sklearn solver,
        use it for tags prediction.
    """
    example = [features]
    return vec.transform(example)

def create_examples(sents, tag_to_idx_dict):
    examples = []
    labels = []
    num_of_sents = 0
    for sent in sents:
        num_of_sents += 1
        for i in range(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tag_to_idx_dict[sent[i][1]])

    return examples, labels


def memm_greedy(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    sent[0] = (sent[0][0],'no_tag')
    ### YOUR CODE HERE
    for i, word in enumerate(sent):
        extracted_feat = extract_features(sent, i)
        predicted_tag_index = logreg.predict(vectorize_features(vec, extracted_feat))[0]
        predict_tag = index_to_tag_dict[predicted_tag_index]
        predicted_tags[i] = predict_tag
        sent[i] = (sent[i][0], predict_tag)
    ### END YOUR CODE
    return predicted_tags

def memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))

    ### YOUR CODE HERE
    def get_prob_features(k, S_t, S_u, sent):
        generated_examples = []
        tags_product_probabilities = []
        tags_product_2index = {}
        extracted_feat = extract_features(sent, k)
        for i, (_t, _u) in enumerate(np.dstack(np.meshgrid(S_t, S_u)).reshape(-1, 2)):
            current_feat = extracted_feat.copy()
            current_feat['trigrams'] = _u + "_" + _t
            current_feat['bigrams'] = _u
            # generated_examples.append(current_feat)
            tags_product_2index[(_t, _u)] = i
            tag_probability = logreg.predict_log_proba(vectorize_features(vec, current_feat))
            tags_product_probabilities.append(tag_probability)
        return tags_product_2index, tags_product_probabilities

    def get_tags_set(word, word_tag_set_dict):
        S = list(word_tag_set_dict[word])
        if len(S) == 0:
            return ['O']
        return S

    pai = defaultdict(dict)
    pai[-1][('*', '*')] = 1
    S = list(index_to_tag_dict.keys())[:-1]
    bp = {}
    n = len(sent)
    for k in range(n):  # word index in sentence
        if k == 0:
            S_t, S_u, S_v = ['*'], ['*'], get_tags_set(sent[k][0], extra_decoding_arguments)
        elif k == 1:
            S_t, S_u, S_v = ['*'], get_tags_set(sent[k-1][0], extra_decoding_arguments), get_tags_set(sent[k][0], extra_decoding_arguments)
        else:
            S_t, S_u, S_v = get_tags_set(sent[k-2][0], extra_decoding_arguments), get_tags_set(sent[k-1][0], extra_decoding_arguments), get_tags_set(sent[k][0], extra_decoding_arguments)

        tags_product_2index, tags_product_probabilities = get_prob_features(k, S_t, S_u, sent)

        for u in S_u:  # previous tag
            for v in S_v:  # current tag
                max_val = -float('Inf')
                best_tag = ""
                for t in S_t:  # previous, previous tag
                    q = tags_product_probabilities[tags_product_2index[(t, u)]][0, tag_to_idx_dict[v]]
                    # t_tag, v_tag = index_to_tag_dict[t], index_to_tag_dict[v]
                    pai_t = pai[k - 1][(t, u)] + q
                    if pai_t > max_val:
                        max_val = pai_t
                        best_tag = t
                pai[k][(u, v)] = max_val
                bp[(k, u, v)] = best_tag

    max_bp_val = -float('Inf')

    for u_, v_ in pai[n - 1]:
        extracted_feat = extract_features(sent, n-1)
        p = pai[n - 1][(u_, v_)]
        if p > max_bp_val:
            u, v = u_, v_
            max_bp_val = p

    if n == 1:
        predicted_tags[n - 1] = v
    else:
        predicted_tags[-1], predicted_tags[-2] = v, u
        for k in range(n - 3, -1, -1):
            predicted_tags[k] = bp[(k + 2, predicted_tags[k + 1], predicted_tags[k + 2])]
    ### END YOUR CODE
    return predicted_tags

def memm_eval(test_data, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
    Receives: test data set and the parameters learned by memm
    Returns an evaluation of the accuracy of Viterbi & greedy memm
    """

    eval_start_timer = time.time()
    gold_tag_seqs = []
    greedy_pred_tag_seqs = []
    viterbi_pred_tag_seqs = []

    # !!! - NEED TO BE DELETED BEFORE SUBMISSION: - !!!
    mistakes = []
    i = 0
    # !!! - NEED TO BE DELETED BEFORE SUBMISSION: - !!!

    for sent in test_data:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        ### YOUR CODE HERE
        prediction_greedy = memm_greedy(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments)
        prediction_viterbi = memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments)
        greedy_pred_tag_seqs.append(prediction_greedy)
        viterbi_pred_tag_seqs.append(prediction_viterbi)

        # Errors Sampling from Viterbi
        if i <= 50:
            mask = np.array(true_tags) != np.array(prediction_viterbi)
            if mask.sum() >= 1:
                mistake_tags = list(zip(np.array(true_tags)[mask], np.array(prediction_viterbi)[mask],
                                        np.array(words)[mask]))
                mistakes.append((i, mistake_tags))
                i += 1
        ### END YOUR CODE

    greedy_evaluation = evaluate_ner(gold_tag_seqs, greedy_pred_tag_seqs)
    viterbi_evaluation = evaluate_ner(gold_tag_seqs, viterbi_pred_tag_seqs)

    # Mistakes log:
    print("Mistakes log:")
    print("-------------")
    for mistake in mistakes:
        index = mistake[0]
        comparisons = mistake[1]
        sentence = [w for (w, t) in test_data[index]]
        print("Sentence: " + str(sentence))
        for comp in comparisons:
            print('token: ' + comp[2])
            print("real value: " + str(comp[0]))
            print("viterbi: " + str(comp[1]))

    return greedy_evaluation, viterbi_evaluation

def build_tag_to_idx_dict(train_sentences):
    curr_tag_index = 0
    tag_to_idx_dict = {}
    for train_sent in train_sentences:
        for token in train_sent:
            tag = token[1]
            if tag not in tag_to_idx_dict:
                tag_to_idx_dict[tag] = curr_tag_index
                curr_tag_index += 1

    tag_to_idx_dict['*'] = curr_tag_index
    return tag_to_idx_dict


if __name__ == "__main__":
    full_flow_start = time.time()
    train_sents = read_conll_ner_file("data/train.conll")
    dev_sents = read_conll_ner_file("data/dev.conll")

    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    extra_decoding_arguments = build_extra_decoding_arguments(train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)
    tag_to_idx_dict = build_tag_to_idx_dict(train_sents)
    index_to_tag_dict = invert_dict(tag_to_idx_dict)

    vec = DictVectorizer()
    print("Create train examples")
    train_examples, train_labels = create_examples(train_sents, tag_to_idx_dict)


    num_train_examples = len(train_examples)
    print("#example: " + str(num_train_examples))
    print("Done")

    print("Create dev examples")
    dev_examples, dev_labels = create_examples(dev_sents, tag_to_idx_dict)
    num_dev_examples = len(dev_examples)
    print("#example: " + str(num_dev_examples))
    print("Done")

    all_examples = train_examples
    all_examples.extend(dev_examples)

    print("Vectorize examples")
    all_examples_vectorized = vec.fit_transform(all_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    print("Done")

    logreg = linear_model.LogisticRegression(
        multi_class='multinomial', max_iter=128, solver='lbfgs', C=100000, verbose=1)
    print("Fitting...")
    start = time.time()
    logreg.fit(train_examples_vectorized, train_labels)
    end = time.time()
    print("End training, elapsed " + str(end - start) + " seconds")
    # End of log linear model training

    # Evaluation code - do not make any changes
    start = time.time()
    print("Start evaluation on dev set")

    memm_eval(dev_sents, logreg, vec, index_to_tag_dict, extra_decoding_arguments)
    end = time.time()

    print("Evaluation on dev set elapsed: " + str(end - start) + " seconds")
