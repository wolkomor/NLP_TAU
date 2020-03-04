from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dateutil.parser import parse
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from empath import Empath
from sklearn.feature_extraction import DictVectorizer
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

def lexical_sentiment_features_extraction(text_tokenized):
    word_count = 0
    number_count = 0
    average_word_length = 0
    #article_length = 0
    count_pos = 0
    count_exclamation = 0
    count_date = 0

    # tokenized = sent_tokenize(text)
    total_sentences = len(text_tokenized)
    for s in text_tokenized:
        s = nltk.word_tokenize(s)
        pos_text = nltk.pos_tag(s)

        for (word, pos) in pos_text:
            word_count += 1
            average_word_length += len(word)
            if word.isdigit():
                number_count += 1
            elif "!" in word:
                count_exclamation += 1
            elif pos=="ADJ":
                count_pos+=1
            try:
                parsed_date = parse(word)
                count_date +=1
            except ValueError:
                try:
                    parsed_date = parse(word, dayfirst=True)
                    count_date += 1
                except ValueError:
                    pass

    average_word_length /=word_count
    return [word_count, number_count, average_word_length, count_pos, count_exclamation, count_date]

def tf_idf(documents_list, gram_range):
    # input is all the corpus documents
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X = tfidf.fit_transform(documents_list)
    return tfidf, X

def categories_extraction(text):
    lexicon = Empath()
    categories = lexicon.analyze(text, normalize=True)
    v = DictVectorizer(sparse=False)
    categories = v.fit_transform(categories)
    return categories

def sentiment_extraction(text):
    polarity_score = SentimentIntensityAnalyzer().polarity_scores(text)
    return polarity_score


def sanity_check():
    text = "Hello my name is OR and born in 18/12/1992 !"
    text2 = "I am beautiful number 1"
    text3 = "I want to hit him"
    docs = [text,text2,text3]
    for doc in docs:
        print (lexical_sentiment_features_extraction(doc))
        print(sentiment_extraction(doc))
        print(categories_extraction(doc))
    tfidf,X = tf_idf(docs, (1,2))
    feature_names = tfidf.get_feature_names()
    rows, cols = X.nonzero()
    for row, col in zip(rows, cols):
        print((feature_names[col], row), X[row, col])

sanity_check()