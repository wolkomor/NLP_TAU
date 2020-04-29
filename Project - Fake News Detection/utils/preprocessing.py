from dateutil.parser import parse
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from empath import Empath
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('punkt')



def lexical_features_extraction(row):

    def allCapsCheck(word):
        return word.isupper() and len(word) > 3

    text = row['text_tokenized']

    word_count = 0
    number_count = 0
    average_word_length = 0
    article_length = 0
    count_pos_adj = 0  # Any kind of adjectives.
    count_pos_sl = 0  # Superlatives adjectives.
    count_exclamation = 0
    count_date = 0
    average_sent_length = 0

    allCapsCount = sum(map(allCapsCheck,  row['text'].split()))

    sentence_tokenized = sent_tokenize(text)
    total_sentences_num = len(sentence_tokenized)

    for s in sentence_tokenized:
        s = nltk.word_tokenize(s)
        average_sent_length += len(s)
        pos_text = nltk.pos_tag(s)

        for (word, pos) in pos_text:
            word_count += 1
            average_word_length += len(word)
            if word.isdigit() or word=='NUMBER':
                number_count += 1
            elif word=='DATE':
                count_date += 1
            elif "!" in word:
                count_exclamation += 1
            elif pos.startswith('JJ'):
                count_pos_adj += 1
                if pos == 'JJS':
                    count_pos_sl += 1
            # else:
            #     try:
            #         parsed_date = parse(word)
            #         count_date += 1
            #     except:
            #         try:
            #             parsed_date = parse(word, dayfirst=True)
            #             count_date += 1
            #         except:
            #             pass

    average_word_length /= word_count
    article_length = average_sent_length
    average_sent_length /= total_sentences_num

    return [article_length, total_sentences_num, average_sent_length, word_count,
            number_count, average_word_length, count_pos_adj,
            count_pos_sl, count_exclamation, count_date, allCapsCount]


def tf_idf(documents_list, gram_range):
    # input is all the corpus documents
    corpus = []
    for example in documents_list:
        corpus.append(token_to_string(example.text))
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X = tfidf.fit_transform(corpus)
    return tfidf, X


def categories_extraction(text, 
                          categories_lst=('law', 'communication', 'crime', 'payment', 'phone', 'banking', 'war',
                                          'economics', 'politics', 'leader', 'social_media', 'school',
                                          'government', 'money', 'work', 'speaking', 'internet', 'business'),
                          if_tokenized=False):
    if if_tokenized:
        text = token_to_string(text)
    lexicon = Empath()
    categories = lexicon.analyze(text, normalize=True, categories=categories_lst)
    if categories==None:
        return [float(0)]*len(categories_lst)
    else:
        return list(categories.values())


def sentiment_extraction(text, if_tokenized=False):
    """Return score to Negative, Neutral, Positive and Compound.
     The Last is aggregated score ranging -1 (mostly neg) to 1 (mostly pos) calculated seperatly"""
    if if_tokenized:
        text = token_to_string(text)
    polarity_score_dict = SentimentIntensityAnalyzer().polarity_scores(text)
    sentiment_names = list(polarity_score_dict.keys())
    polarity_score_array = np.fromiter(polarity_score_dict.values(), dtype=float)
    if polarity_score_dict == None:
        return [float(0)]*4
    else:
        return list(polarity_score_dict.values())


def token_to_string(tokenized_words_lst):
    return ' '.join(tokenized_words_lst[1:-1])


def sanity_check(data=False):

    if not data:
        text = "Hello my name is OR and born in 18/12/1992 !"
        text2 = "I am beautiful number 1"
        text3 = "I want to hit him"
        docs = [text, text2, text3]
    else:
        docs = data.examples

    metadata_features = np.zeros((1, 208))

    for doc in docs:
        print('lexical_sentiment_features_extraction')
        scores_lex = lexical_features_extraction(token_to_string(doc.text))
        print(scores_lex)
        print('\nsentiment_extraction')
        scores_sentiment = sentiment_extraction(doc.text, if_tokenized=True)
        print(scores_sentiment)
        print('\ncategories_extraction')
        scores_cat = categories_extraction(doc.text, if_tokenized=True)
        print(scores_cat)
        doc_features = np.concatenate([np.array(scores_lex), np.array(scores_sentiment), scores_cat.reshape((-1))])
        metadata_features = np.concatenate((metadata_features, doc_features.reshape((1, -1))), axis=0)

    metadata_features = metadata_features[1:, :]
    tfidf, X = tf_idf(docs, (1, 2))
    feature_names = tfidf.get_feature_names()


    # Print the results:
    rows, cols = X.nonzero()
    for row, col in zip(rows, cols):
        print((feature_names[col], row), X[row, col])

    return metadata_features


if __name__ == "__main__":
    sanity_check()