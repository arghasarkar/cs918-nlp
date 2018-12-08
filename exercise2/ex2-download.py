#!/usr/bin/env python
# coding: utf-8

# # NLTK Path
#
# The code cell below is used for setting the NLTK path on DCS machines and Joshua. On my personal computer where this assignment was done, I had downloaded the NLTK corpus using **nltk.download()**
#

# In[1]:


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
FILE_NAME = "./twitter-dev-data.txt"

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

            #             if _score > 0:
            #                 self.pos_words[_word] = _score
            #             else:
            #                 self.neg_words[_word] = _score

            self.all_words[_word] = _score

    def print_words_summary(self):
        print(
        "Positive count: {} Negative count: ".format(str(len(self.pos_words.keys()))), str(len(self.neg_words.keys())))


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

tokenizer = TweetTokenizer()


class Tweet:

    def __init__(self, raw_str):
        self.raw_str = raw_str
        self.id, self.sentiment, self.tweet = self.parse_raw_tweet()

        self.parsed_tweet = ""
        self.tokens = []

        # Methods to run
        self.preprocess_tweet()

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

    def tokenize(self):
        self.tokens = tokenizer.tokenize(self.tweet)

    def __str__(self):
        return ("Tweet ID: {} -- Sentiment: {} -- Tweet: {} \n Tokens: {}\n".format(str(self.id), self.sentiment,
                                                                                    self.tweet, self.tokens))


# # '''
# # Test code below. IGNORE -------
# # '''
# rt1 = "735752723159607191	positive	Shay with Bentley and Bella, in their sunday best https://google.co.uk :) http://t.co/SUMZBSTrkW hello"
# rt2 = "529243425878060644	negative	Dear MSM, CNN bitches, every election one turning point..there you go again I will not use my opponents youth & NOW \"basket of deplorables\""
# rt3 = "348472267247705036	negative	@LifeNewsHQ CHIP defines a child at conception. Some Democrats want to end CHIP by folding it into Medicaid. Should… https://t.co/To21fCSHkO"

# t1 = Tweet(rt1)
# t2 = Tweet(rt2)
# t3 = Tweet(rt3)

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

    def __init__(self, file_path):
        # Private vars
        self._corpus_loaded = False

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
            _tweet = Tweet(line)
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

# In[ ]:


# In[6]:


import pprint


class X:
    def __init__(self, val):
        self.val = val

    def get_val(self):
        return self.val

    def set_val(self, val):
        self.val = val

    def __str__(self):
        return "Value: {}".format(str(self.val))


dic = {}
new = {}

x = X(0)
y = X(2)

dic["0"] = x
dic["1"] = y

new["0"] = x
new["1"] = y

for key in dic:
    print("Key: {} Val: {}".format(str(key), str(dic[key])))

a = dic["0"]
b = dic["1"]

a.set_val(100)
b.set_val(1001)

print("printing from old dict:")
some = dic["0"]
some.val = "Argha"

for key in dic:
    print("Key: {} Val: {}".format(str(key), str(dic[key])))

print("printing from new dict:")

for key in new:
    print("Key: {} Val: {}".format(str(key), str(new[key])))


# # Lexicon classifier class
#
# This classifier uses the word's positive and negative ratings to work out the overall sentiment. The method ```classify_tweet(Tweet)``` takes the Tweet class as an argument. It extracts the Tweet's text and then classifies it.

# In[19]:


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

        sentiment = "neutral"

        return score, str(sentiment)


# Declaring the lexicon Classifier
lc = LexiconClassifier(ws)


# print(lc.classify_tweet(Tweet("382489758445350006	negative	high prozac")))


# # Test loader class
#
# Class for loading the test sets. It accepts the **file_name** and the **classifier** as arguments. It returns the dictionary of the predictions.

# In[20]:


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
            results_dict[str(_id)] = classification[1]

        return results_dict


# td1 = TestData(testsets.testsets[0], c, lc)
# td1.run_classifier()


# In[ ]:


# In[ ]:


# In[29]:


for classifier in ['lex_classifier', 'myclassifier2',
                   'myclassifier3']:  # You may rename the names of the classifiers to something more descriptive

    valFound = False

    if classifier == 'lex_classifier':
        print('Training ' + classifier)
        # TODO: extract features for training classifier1
        # TODO: train sentiment classifier1
    elif classifier == 'myclassifier2':
        print('Training ' + classifier)
        # TODO: extract features for training classifier2
        # TODO: train sentiment classifier2
    elif classifier == 'myclassifier3':
        print('Training ' + classifier)
        # TODO: extract features for training classifier3
        # TODO: train sentiment classifier3

    for testset in testsets.testsets:
        # TODO: classify tweets in test set

        #         predictions = {'163361196206957578': 'neutral', '768006053969268950': 'neutral', '742616104384772304': 'neutral', '102313285628711403': 'neutral', '653274888624828198': 'neutral'} # TODO: Remove this line, 'predictions' should be populated with the outputs of your classifier

        test_corpus = Corpus(testset)
        test_corpus.load_corpus()
        test_corpus.parse_corpus()
        #         test_corpus.print_summary_of_corpus()

        td1 = TestData(testset, test_corpus, lc)
        predictions = td1.run_classifier()
        print(predictions)
        #         predictions = {}

        #         if '163361196206957578' in predictions:
        #             print("--------------------------------VAL FOUND-----------------------------------")

        evaluation.evaluate(predictions, testset, classifier)

        evaluation.confusion(predictions, testset, classifier)

# In[10]:


tok = TweetTokenizer()

t1 = "\"#Obergefell, Marriage Equality and Islam in the West http://t.co/NoQlB3g6t0 #IslaminAmerica #marriageequality\""
t2 = "@LifeNewsHQ CHIP defines a child at conception. Some Democrats want to end CHIP by folding it into Medicaid. Should… https://t.co/To21fCSHkO"

tok.tokenize(t2)

# In[ ]:




