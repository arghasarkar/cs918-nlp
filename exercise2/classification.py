
import getpass

username = getpass.getuser()

if username is not "ArghaWin10" or username is not "arghasarkar":
    import nltk
    from nltk.tokenize import TweetTokenizer

    nltk.data.path.append('/modules/cs918/nltk_data/')

# # Sanity check
#
# Sanity check to ensure that the file exists before proceeding further

# In[2]:


# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import testsets
import evaluation

from os.path import normpath
import os

# TODO: load training data
FILE_NAME = "./twitter-training-data.txt"

if os.path.isfile(FILE_NAME):
    print("The file has been found.")
else:
    raise IOError("File not found.")

# # Load sentiment marked lexicons
#
# Using lexicons marked with a value indicating positive / neutral / negative sentiment. These words and their counts in tweets will be used for checking the sentiment of the tweets. This is only as a basic classifier.
#
# The format of the file is like:
#
# **word{tab}sentiment_value**
#
# The sentiment value is between +1 to -1 and are real numbers.
#
# - +1 is extremely positive
# - -1 is extremely negative
# - Everything in between is on the postive / negative spectrum.
#
# Using a dictionary where the **word is the key** and **the value is their rating**.

# In[3]:


WORDS_FILE_NAME = "wordsWithStrength.txt"


class WordSentiment:

    def __init__(self, file_name):

        self.file_name = file_name

        self.pos_words = {}
        self.neg_words = {}
        self.all_words = {}

        self.load_words()

    def load_words(self):

        if os.path.isfile(FILE_NAME):
            print("The words sentiment file has been found.")
        else:
            raise IOError("File not found: Words Sentiment.")

        with open(self.file_name, "r") as corpus:
            data = corpus.readlines()
            self.process_words(data)

    def process_words(self, data):

        for line in data:
            split_line = line.split("\t")
            _word = split_line[0].lower()
            _score = split_line[1]
            _score = float(_score)

            self.all_words[_word] = _score

    def print_words_summary(self):
        print("Positive count: {} Negative count: ".format(str(len(self.pos_words.keys()))),
              str(len(self.neg_words.keys())))


ws = WordSentiment(WORDS_FILE_NAME)
ws.print_words_summary()

# # Tweet class
#
# The Tweet class is passed on a single line from the training dataset. It then parses it to get the ```tweet_id```, sentiment and the actual tweet.
#
# Additionally, it is also responsible for performing the preprocessing required during the instantiation.
#

# In[4]:


import re

from nltk.stem import WordNetLemmatizer

tokenizer = TweetTokenizer()


class Tweet:

    def __init__(self, raw_str, lemmatizer):
        self.raw_str = raw_str
        self.id, self.sentiment, self.tweet = self.parse_raw_tweet()

        self.parsed_tweet = ""
        self.tokens = []

        self.lemmatizer = lemmatizer

        # Methods to run
        self.preprocess_tweet()

        # Post tokenized string
        self.tok_str = ""
        # Tokenize preprocessed tweet
        self.tokenize()

    def parse_raw_tweet(self):
        parts = self.raw_str.split("\t")
        return parts[0], parts[1], parts[2]

    def preprocess_tweet(self):
        # Lower case the string
        self.tweet = self.tweet.lower()

        # Replace URLs
        self._preprocess_replace_URLs()

        # Replace user mentions
        self._preprocess_replace_user_mentions()

        # Remove one character long words
        self._preprocess_remove_one_char_long_words()

        # Substitute emojis
        self._preprocess_substitute_emojis()

        # Lemmatize
        self._preprocess_lemmatize()

        # Remove non-alphanumeric characters
        self._preprocess_remove_all_non_alphanumeric_chars()

    def _preprocess_replace_URLs(self):
        self.tweet = re.sub(r'\b(http)(s)?:\/\/((([\w\/])(\.)?)+)\b', 'urllink', self.tweet)

    def _preprocess_replace_user_mentions(self):
        self.tweet = re.sub(r'^(?!.*\bRT\b)(?:.+\s)?@\w+', 'usermention', self.tweet)

    def _preprocess_remove_one_char_long_words(self):
        self.tweet = re.sub(r'\b[A-Za-z0-9]{1}\b', ' ', self.tweet)

    def _preprocess_substitute_emojis(self):
        self.tweet = re.sub(r':\)|:]|:3|:>|8\)|\(:|=\)|=]|:\'\)', 'happyface', self.tweet)
        self.tweet = re.sub(r':\(|:\[|:<|8\(|\(:|=\(|=\[|:\'\(|:-\(', 'sadface', self.tweet)

    def _preprocess_remove_all_non_alphanumeric_chars(self):
        self.tweet = re.sub(r'[^\sa-zA-Z0-9]', ' ', self.tweet)

    def _preprocess_lemmatize(self):
        lemmatizer = self.lemmatizer
        words = self.tweet.split()
        for i in range(len(words)):
            words[i] = lemmatizer.lemmatize(words[i])

        self.tweet = " ".join(words)

    def tokenize(self):
        self.tokens = tokenizer.tokenize(self.tweet)
        self.tok_str = " ".join(self.tokens)

    def __str__(self):
        return ("Tweet ID: {} -- Sentiment: {} -- Tweet: {} \n Tokens: {}\n".format(str(self.id), self.sentiment,
                                                                                    self.tok_str, self.tokens))


# # '''
# # Test code below. IGNORE -------
# # '''

wnLemmatizer = WordNetLemmatizer()

# rt1 = "735752723159607191	positive	Shay with Bentley and Bella, in their sunday best https://google.co.uk :) http://t.co/SUMZBSTrkW hello"
# rt2 = "529243425878060644	negative	Dear MSM, CNN bitches, every election one turning point..there you go again I will not use my opponents youth & NOW \"basket of deplorables\""
# rt3 = "348472267247705036	negative	@LifeNewsHQ CHIP defines a child at conception. Some Democrats want to end CHIP by folding it into Medicaid. Should… https://t.co/To21fCSHkO"

# t1 = Tweet(rt1, wnLemmatizer)
# t2 = Tweet(rt2, wnLemmatizer)
# t3 = Tweet(rt3, wnLemmatizer)

# print(t1)
# print(t2)
# print(t3)


# # Corpus
#
# Corpus object to read the data and preprocess it

# In[5]:


import json
import itertools

from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize


class Corpus:

    def __init__(self, file_path, lemmatizer=WordNetLemmatizer()):
        # Private vars
        self._corpus_loaded = False
        self.lemmatizer = lemmatizer

        self.file_path = file_path

        # Checks if the file exists
        if os.path.isfile(self.file_path):
            print("The file has been found.")
        else:
            raise IOError("File not found.")

        self.raw_docs = []

        # Contains all the tweets. Key: Tweet ID, Value: Tweet object
        self.processed_dict = {}

        # Contains all the positive tweets. Key: Tweet ID, Value: Tweet object
        self.pos_tweets = {}

        # Contains all the neutral tweets. Key: Tweet ID, Value: Tweet object
        self.neu_tweets = {}

        # Contains all the negativve tweets. Key: Tweet ID, Value: Tweet object
        self.neg_tweets = {}

    def load_corpus(self):

        with open(self.file_path, "r", encoding="utf8") as corpus:
            data = corpus.readlines()
            for line in data:
                self.raw_docs.append(line)

        self._corpus_loaded = True

    def parse_corpus(self):

        for line in self.raw_docs:
            _tweet = Tweet(line, self.lemmatizer)
            _sentiment = _tweet.sentiment
            _id = _tweet.id

            if _sentiment == "positive":
                self.pos_tweets[_id] = _tweet
            elif _sentiment == "neutral":
                self.neu_tweets[_id] = _tweet
            else:
                self.neg_tweets[_id] = _tweet

            self.processed_dict[_id] = _tweet

        self._corpus_parsed = True

    def print_summary_of_corpus(self):

        if not self._corpus_loaded:
            self.load_corpus()

        if not self._corpus_parsed:
            self.parse_corpus()

        print("Number of training samples: {}.".format(str(len(self.raw_docs))))
        print("Number of positive samples: {}.".format(str(len(self.pos_tweets))))
        print("Number of neutral samples: {}.".format(str(len(self.neu_tweets))))
        print("Number of negative samples: {}.".format(str(len(self.neg_tweets))))

    def print_positive_tweets(self):

        for _id in self.pos_tweets:
            print(self.pos_tweets[_id])

    def print_neutral_tweets(self):

        for _id in self.neu_tweets:
            print(self.neu_tweets[_id])

    def print_negative_tweets(self):

        for _id in self.neg_tweets:
            print(self.neg_tweets[_id])


c = Corpus(FILE_NAME)
c.load_corpus()
c.parse_corpus()
c.print_summary_of_corpus()

# # Building a count vector
#
# For the second classifier, a count vector is required. Before that can be done, a list of all tweets is needed. So first all, need to go through all the tweets in the corpus and build a new list of just tweet strings. After building the list of tweet strings, use ```CountVectorizer()``` to transform it into a vector by using the ```vectorizer.fit_transform(corpus)``` function. Whilst doing this, we need to also keep track of the labels of each tweet.

# In[6]:


from sklearn.feature_extraction.text import CountVectorizer


class CountVectorCorpus:

    def __init__(self, original_corpus):
        self.vectorizer = CountVectorizer()

        self.original_corpus = original_corpus

        self.tweets_list, self.labels_list, self.vector_corpus = self.build_list()

    def build_list(self):
        tweets = self.original_corpus.processed_dict

        x_list = []
        y_list = []
        for _id in tweets:
            tweet = tweets[_id]
            _sentiment = tweet.sentiment
            x_list.append(tweet.tok_str)
            y_list.append(_sentiment)

        vect_corp = self.vectorizer.fit_transform(x_list)
        return x_list, y_list, vect_corp


cvc = CountVectorCorpus(c)
# print(cvc.vector_corpus[0])


# # TF IDF
#
#

# In[7]:


# from sklearn.feature_extraction.text import TfidfTransformer

# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(cvc.vector_corpus)
# X_train_tfidf.shape


# # Naive bayes

# In[8]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer


class MultinomialNaiveBayesClassifier:

    def __init__(self, count_vector):

        # Flag to see if model has been trained
        self.trained = False

        self.x_train_tfidf = None
        self.count_vector = count_vector
        self.tfidf_transformer = None

        self.model = None

    def train(self):
        self.tfidf_transformer = TfidfTransformer()
        self.x_train_tfidf = self.tfidf_transformer.fit_transform(self.count_vector.vector_corpus)

        self.model = MultinomialNB().fit(self.x_train_tfidf, self.count_vector.labels_list)
        self.trained = True

    def classify_tweet(self, tweet):

        document = tweet.tweet
        document = [document]

        if self.trained == False:
            self.train()

        tfidf_transformer = TfidfTransformer()

        x_new_counts = self.count_vector.vectorizer.transform(document)
        x_new_tfidf = self.tfidf_transformer.transform(x_new_counts)

        predicted = self.model.predict(x_new_tfidf)

        for doc, category in zip(document, predicted):
            return str(category)


naiveBayesClassifier = MultinomialNaiveBayesClassifier(cvc)
naiveBayesClassifier.train()
naive_bayes_model = naiveBayesClassifier.model

docs_new = Tweet(
    '071288451742262774	negative	Missed @atmosphere at Soundset due to tornado. Now they are going to in DSM tomorrow. Do i want to put up with crowds and spend $45 more?',
    wnLemmatizer)

result = naiveBayesClassifier.classify_tweet(docs_new)
print(result)

# clf = MultinomialNB().fit(X_train_tfidf, cvc.labels_list)


# # Glove word embedding
#
# Glove word embedding's Twitter dataset is being used. From that, the file called ```glove.twitter.27B.25d.txt``` in particular is being used.
#
# The **GloveTwitterWordEmbedding** class is responsible for loading up the file and parsing the dataset. It'll also be responsible for converting each word and sentence into a vector and list of vectors that can then be passed onto the most appropriate machine learning algorithm.
#
# After the embeddings have been loaded and parsed, ```get_embeddings_for_sentence(self, sentence)``` can be used for passing on a sentence and getting the embeddings value. In this instance, **sum** function has been used to convert the embeddings for every single word in the sentence into a single vector represening the whole sentence.

# In[9]:


import numpy as np

GLOVE_FILE_NAME = "glove.twitter.27B.25d.txt"


class GloveTwitterWordEmbedding:

    def __init__(self, file_name):

        self.file_name = file_name
        self.file_data = None

        if os.path.isfile(file_name):
            print("The Glove Word Embedding file has been found.")
        else:
            raise IOError("File not found: Glove Twitter Embedding file with filename: {}".format(file_name))

        self.embeddings = {}

        self.file_loaded = False
        self.file_parsed = False

        self.process_file()

    def load_file(self):

        with open(self.file_name, "r", encoding="utf8") as f:
            self.file_data = f.readlines()
            self.loaded_file = True

    def process_file(self):

        if self.file_loaded == False:
            self.load_file()

        for i in range(len(self.file_data)):
            _emb = self.file_data[i]
            _emb_parts = _emb.split()

            # Storing the word embeddings in a dictionary where the key is the word and the value is
            # a numpy array of the word embedding vector.
            self.embeddings[str(_emb_parts[0])] = np.array(_emb_parts[1:], dtype=np.float32)

        self.file_parsed = True

    def get_embeddings_for_sentence(self, sentence):

        _embeddings = []

        _words = sentence.split()
        for i in range(len(_words)):

            _embedding = np.array(
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                dtype=np.float32)
            if str(_words[i]) in self.embeddings:
                _embedding = self.embeddings[str(_words[i])]

            _embeddings.append(_embedding)

        _embeddings = np.array(_embeddings, dtype=np.float32)
        _emb_sum = np.sum(_embeddings, axis=0, dtype=np.float32)

        return _emb_sum


# In[29]:


word_embedding.get_embeddings_for_sentence("hello world")

# # Word embedding, SVM classifier
#
# Using SVM with word embedding as the third (final) classifier.

# In[23]:


from sklearn import svm
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier


class SVMClassifier:

    def __init__(self, original_corpus, word_embedding):

        self.original_corpus = original_corpus
        self.word_embedding = word_embedding

        self.target_dict = {
            "negative": 1,
            "neutral": 2,
            "positive": 3
        }

        self.reverse_target_dict = {
            1.: "negative",
            2.: "neutral",
            3.: "positive"
        }

        self.train_x = []
        self.train_y = []

        self.model = None

        self.prepare_data()

    def prepare_data(self):

        _tweets = self.original_corpus.processed_dict

        for _id in _tweets:
            _tweet = _tweets[_id]
            _sentiment = _tweet.sentiment
            _tweet_str = _tweet.tweet

            _tweet_embedding_vector = self.word_embedding.get_embeddings_for_sentence(_tweet_str)

            self.train_x.append(_tweet_embedding_vector)
            self.train_y.append(self.target_dict[_sentiment] if _sentiment in self.target_dict else 0)

    def train(self):

        import pickle

        my_svm_model_name = "my_svm_model.pkl"

        if os.path.isfile(my_svm_model_name):
            # Files exists so read it
            print("Reading pickle")
            with open(my_svm_model_name, 'rb') as f:
                self.model = pickle.load(f)
        else:
            # Create it and save it

            self.model = OneVsRestClassifier(estimator=SVC(gamma='auto', random_state=0))
            self.model.fit(self.train_x, self.train_y)

            print("writing pickle")
            with open(my_svm_model_name, 'wb') as f:
                pickle.dump(self.model, f)

    def classify_tweet(self, tweet):

        _tweet_str = tweet.tweet
        _embedding = self.word_embedding.get_embeddings_for_sentence(_tweet_str)

        prediction = self.model.predict(_embedding)

        return self.reverse_target_dict[prediction[0]]


# In[24]:


# my_svm.train()


# In[25]:


# print(my_svm.classify_tweet(Tweet("910335772112060797	positive	My moms under the hilarious impression that I'm spending my saturday loading & unloading furniture from trailers in natchitoches", wnLemmatizer)))


# In[27]:


# print(my_svm.model)


# In[28]:


# import pickle

# my_svm_model_name = "my_svm_model.pkl"

# my_model = None

# if os.path.isfile(my_svm_model_name):
#     # Files exists so read it
#     print("Reading pickle")
#     with open(my_svm_model_name, 'rb') as f:
#         my_model = pickle.load(f)
# else:
#     # Create it and save it
#     print("writing pickle")
#     my_model = my_svm.model
#     with open(my_svm_model_name, 'wb') as f:
#         pickle.dump(my_model, f)


# # Lexicon classifier class
#
# This classifier uses the word's positive and negative ratings to work out the overall sentiment. The method ```classify_tweet(Tweet)``` takes the Tweet class as an argument. It extracts the Tweet's text and then classifies it.

# In[16]:


class LexiconClassifier:

    def __init__(self, word_sentiment):

        self.min_neut = -0.15
        self.max_neut = 0.15

        self.word_sentiment = word_sentiment

    def classify_tweet(self, tweet):

        words_dict = self.word_sentiment.all_words
        score = float(0)
        tokens = tweet.tokens

        sentiment = ""

        for tok in tokens:
            if tok in words_dict:
                score += words_dict[tok]

        if score > self.min_neut and score < self.max_neut:
            sentiment = "neutral"
        elif score > self.max_neut:
            sentiment = "positive"
        else:
            sentiment = "negative"

        return str(sentiment)


# Declaring the lexicon Classifier
lc = LexiconClassifier(ws)


# print(lc.classify_tweet(Tweet("382489758445350006	negative	high prozac")))


# # Test loader class
#
# Class for loading the test sets. It accepts the **file_name** and the **classifier** as arguments. It returns the dictionary of the predictions.

# In[17]:


class TestData:

    def __init__(self, file_name, corpus, classifier):
        self.file_name = file_name
        self.corpus = corpus
        self.classifier = classifier

        self.classified_dict = {}

    def run_classifier(self):
        # Predictions dictionary
        results_dict = {}

        # Tweets dict
        td = self.corpus.processed_dict

        for _id in td:
            _tweet = td[_id]
            classification = self.classifier.classify_tweet(_tweet)
            results_dict[str(_id)] = classification

        return results_dict


# # Main classification
#
# The code below is part of the skeleton code that was provided in ```classification.py``` file.

# In[33]:


import warnings

warnings.filterwarnings('ignore')

for classifier in ['lex_classifier', 'naive_bayes',
                   'svm']:  # You may rename the names of the classifiers to something more descriptive

    # Used for Multinomial Naive Bayes classifier
    count_vector = CountVectorCorpus(c)
    naiveBayesClassifier = MultinomialNaiveBayesClassifier(count_vector)

    # Used for word_embeddings and SVM classifier
    word_embedding = GloveTwitterWordEmbedding(GLOVE_FILE_NAME)
    my_svm = SVMClassifier(c, word_embedding)

    if classifier == 'lex_classifier':
        print('Training is NOT required for: ' + classifier)

    elif classifier == 'naive_bayes':
        print('Training ' + classifier)
        naiveBayesClassifier.train()

    elif classifier == 'svm':
        print('Training ' + classifier)
        # TODO: extract features for training classifier3
        # TODO: train sentiment classifier
        my_svm.train()

    for testset in testsets.testsets:
        # TODO: classify tweets in test set

        #         predictions = {'163361196206957578': 'neutral', '768006053969268950': 'neutral', '742616104384772304': 'neutral', '102313285628711403': 'neutral', '653274888624828198': 'neutral'} # TODO: Remove this line, 'predictions' should be populated with the outputs of your classifier
        predictions = {}

        if classifier == 'lex_classifier':
            test_corpus = Corpus(testset)
            test_corpus.load_corpus()
            test_corpus.parse_corpus()

            td1 = TestData(testset, test_corpus, lc)
            predictions = td1.run_classifier()

        elif classifier == 'naive_bayes':
            test_corpus = Corpus(testset)
            test_corpus.load_corpus()
            test_corpus.parse_corpus()

            td2 = TestData(testset, test_corpus, naiveBayesClassifier)
            predictions = td2.run_classifier()

        elif classifier == 'svm':

            test_corpus = Corpus(testset)
            test_corpus.load_corpus()
            test_corpus.parse_corpus()

            td3 = TestData(testset, test_corpus, my_svm)
            predictions = td3.run_classifier()

        # Evaluation
        evaluation.evaluate(predictions, testset, classifier)
        evaluation.confusion(predictions, testset, classifier)


