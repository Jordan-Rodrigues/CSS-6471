from scipy import stats
import numpy as np
import consts
import pandas as pd
from nltk.tokenize import word_tokenize
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import string
import gensim
from gensim.corpora import Dictionary
from gensim.models import Phrases
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS


class LDADF:
    def __init__(self, df):
        """
        Initializes class

        Parameters
        ----------
        df: pd.DataFrame
            the dataframe to analyze (end result of utils.py)

        Returns
        -------
        None
        """
        self.df = df[~df["parsed_tweets"].isna()]

    def create_dic_and_corp(self):
        """
        Given the parsed text data of utils, tokenize it, combine with hashtags, and create dict/corpus

        Parameters
        ----------
        None

        Returns
        -------
        dic: dictionary that maps indexes to words
        corpus: tdf for each word in each document, for all docs
        """
        # create extra stopwords for removal
        ex_stop = (
            "olympic",
            "olympics",
            "i",
            "s",
            "",
            "the",
        )

        words = self.df["parsed_tweets"].values
        hashtags = self.df["hashtags"].values
        i = 0
        tokenized = [0] * len(words)
        for tweet, tags, i in zip(words, hashtags, range(len(words))):
            # error handling for empty tweetsa
            tags = tags.strip("][").split(", ")
            tags = [
                x[2:-1].lower() for x in tags
            ]  # get rid of quote symbol + hashtag + lower
            try:
                tweet = tweet.translate(
                    str.maketrans("", "", string.punctuation)
                ).lower()  # remove punctuation and lowercase

                # parse tags
                tweet = word_tokenize(tweet) + tags
            except Exception as e:
                tweet = tags

            # Remove numbers, but not words that contain numbers.
            tweet = [x for x in tweet if not x.isnumeric()]

            # Remove words that are only one character.
            tweet = [x for x in tweet if len(x) > 1]

            # Remove non-ascii
            tweet = [x.encode("ascii", "ignore").decode() for x in tweet]

            # Remove stopwords
            tweet = [x for x in tweet if x not in STOPWORDS]

            # Remove extra stopwords
            tokenized[i] = [x for x in tweet if x not in ex_stop]

        # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
        bigram = Phrases(tokenized, min_count=100)
        for idx in range(len(tokenized)):
            for token in bigram[tokenized[idx]]:
                if "_" in token:
                    # Token is a bigram, add to document.
                    tokenized[idx].append(token)

        # Create Dictionary
        dic = corpora.Dictionary(tokenized)
        dic.filter_extremes(
            no_below=100, no_above=0.6
        )  # total is roughly 400K -- set no above greater than any individual proportion (want to keep beijing/tokyo) and no below to 75 tweets
        print("dic size:", len(dic))
        # Term Document Frequency
        corpus = [dic.doc2bow(tweet) for tweet in tokenized]
        return dic, corpus, tokenized
