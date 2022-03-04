import pandas as pd
import fasttext
import re
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from nltk import tokenize 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def get_merged(file_1: str, file_2: str, exists=False) -> pd.DataFrame:
    """
    Get merged datasets. Combines the two tweet datasets by distinct tweet_id and drops duplicates, or pulls from file if available.

    Parameters
    ----------
    file_1 : str
        First csv name
    file_2 : pd.DataFrame
        Second csv name
    exists : bool, optional
        Flag on whether to generate merge or read from file

    Returns
    -------
    merged_df
        combined dataframe
    """
    if exists:
        return pd.read_csv('../data/merged.csv')
    df_1 = pd.read_csv(file_1)
    df_2 = pd.read_csv(file_2)

    #clean id column, cast to int
    int_id = pd.to_numeric(df_1.id, errors='coerce')
    bad_ids = int_id.isna()
    df_1 = df_1[~bad_ids]
    df_1['id'] = df_1['id'].astype(int)
    df_1.drop_duplicates(subset=['id'], inplace=True)
    df_1 = drop_na(df_1)

    int_id = pd.to_numeric(df_2.id, errors='coerce')
    bad_ids = int_id.isna()
    df_2 = df_2[~bad_ids]
    df_2['id'] = df_2['id'].astype(int)
    df_2.drop_duplicates(subset=['id'], inplace=True)
    df_2 = drop_na(df_2)
    
    #drop columns unique to either set
    df_1.drop(axis=1, columns=['hashtags', 'user_verified', 'source', 'is_retweet', 'user_favourites'], inplace=True)
    df_2.drop(axis=1, columns=['favorited','language'], inplace=True)
    
    #rename columns to match
    df_2.rename(columns={'user_screen_name':'user_name', 'retweet_count': 'retweets', 'favorite_count':'favorites', 'user_created_at':'user_created'}, inplace=True)


    #drop dups
    ids = set(df_1['id'].values)
    df_2 = df_2[~df_2['id'].isin(ids)]
    merged = pd.concat([df_1, df_2])
    merged.set_index(['id'], inplace=True, verify_integrity=True)
    merged.reset_index(inplace=True) # keep id in a non index column

    #to csv
    merged.to_csv('../data/merged.csv')
    return merged

def merge_years(file_1: str, file_2: str, exists=False) -> pd.DataFrame:
    """
    Get merged datasets. Combines the two olympics datasets (2020 vs 2022), or pulls from file if available.

    Parameters
    ----------
    file_1 : str
        First csv name
    file_2 : pd.DataFrame
        Second csv name
    exists : bool, optional
        Flag on whether to generate merge or read from file

    Returns
    -------
    merged_df
        combined dataframe
    """

    if exists:
        return pd.read_csv('../data/merged_years.csv')
    
    df_1 = pd.read_csv(file_1)
    df_2 = pd.read_csv(file_2)

    #drop cols unique to either one
    df_2.drop(axis=1, columns=['lang', 'public_metrics.reply_count', 'public_metrics.quote_count', 'author.id', 'author.public_metrics.tweet_count', 'author.name'], inplace=True)

    df_2.rename(columns={'author.location':'user_location', 'author.description': 'user_description', 'author.created_at':'user_created', 'created_at':'date', 'author.public_metrics.followers_count': 'user_followers', 'author.public_metrics.following_count': 'user_friends', 'public_metrics.retweet_count':'retweets', 'public_metrics.like_count':'favorites', 'author.username':'user_name'}, inplace=True)

    #fix dtypes of first df
    df_2['user_followers'] = df_2['user_followers'].astype(float)
    df_2['user_friends'] = df_2['user_friends'].astype(float)
    df_2['retweets'] = df_2['retweets'].astype(float)
    df_2['favorites'] = df_2['favorites'].astype(float)

    merged = pd.concat([df_1, df_2])
    
    merged.to_csv('../data/merged_years.csv', index=False)
    return merged

def drop_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataset and removes the rows that have column values for 'text' that are null

    Parameters
    ----------
    df: pd.DataFrame
        the df to be parsed

    Returns
    -------
    df
        df, but only with non-null text rows
    """
    return df[~df['text'].isna()]


def get_english(df: pd.DataFrame, exists=False) -> pd.DataFrame:
    """
    Takes a dataset and removes the rows that have column values for 'text' that are non-English

    Parameters
    ----------
    df: pd.DataFrame
        the df to be parsed
    exists : bool, optional
        Flag on whether to generate eng or read from file
    

    Returns
    -------
    english_df
        df, but only with english rows
    """
    if exists:
        return pd.read_csv('../data/english.csv')

    PRETRAINED_MODEL_PATH = '../data/lid.176.bin'
    model = fasttext.load_model(PRETRAINED_MODEL_PATH)
    text_arr = df['text'].values
    mask = []

    for text in text_arr: 

        #clean text
        text = re.sub("@[A-Za-z0-9_]+","", text)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r'\n','', text)
        text = re.sub(r'\r','', text)

        pred = model.predict(text)
        if pred[0] == ('__label__en',):
            mask.append(1)
        else:
            mask.append(0)
    mask = np.array([mask])
    mask = np.array(mask, dtype=bool)
    english = df[mask[0]]

    #to csv
    english.to_csv('../data/english.csv')
    return english

def get_sentiment(df, exists=False, downloads=False):
    """
    Helper function that takes a dataframe with a text column, and adds a sentiment score column using NLTK

    Parameters
    ----------
    df: pd.DataFrame
        the dataframe to be updated
    exists : bool, optional
        Flag on whether to generate eng or read from file
    downloads: boolean
        set downloads = true for the first time to download nltk packages

    Returns
    -------
    df: pd.DataFrame
        dataframe with a new column for sentiment
    """
    if exists:
        return pd.read_csv('../data/sentiment.csv')

    if downloads:
        nltk.download('punkt')
        nltk.download('vader_lexicon')
        nltk.download('stopwords')
    tweets = df['text']
    scores = [-1] * len(tweets)
    sid = SentimentIntensityAnalyzer()
    for tweet, i in zip(tweets, range(len(tweets))):
        #clean tweet
        tweet = re.sub(r'\n','', tweet)
        tweet = re.sub(r'\r','', tweet)
        tweet = re.sub("@[A-Za-z0-9_]+","", tweet)
        tweet = re.sub(r"http\S+", "", tweet)
        ss = sid.polarity_scores(tweet)['compound']
        scores[i] = ss
    df['sentiment'] = scores
    df.to_csv('../data/sentiment.csv', index=False)
    return df
    
def parse_tweets(df, exists=False, downloads=False):
    """
    Helper function that takes a dataframe with a text column, and adds a parsed version of the tweet to the dataset

    Parameters
    ----------
    df: pd.DataFrame
        the dataframe to be updated
    exists : bool, optional
        Flag on whether to generate eng or read from file
    downloads: boolean
        set downloads = true for the first time to download nltk packages
    Returns
    -------
    df: pd.DataFrame
        dataframe with new columns for parsed text and hashtags
    """
    if exists:
        return pd.read_csv('../data/parsed.csv')

    if downloads:
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    cached_stopwords = stopwords.words("english")
    tweets = df['text'].values
    parsed_tweets = [""] * len(tweets)
    hashtag_list = [None] * len(tweets)
    df['parsed_tweets'] = ""
    df['hashtags'] = np.empty((len(df), 0)).tolist()
    sid = SentimentIntensityAnalyzer()
    for tweet, i in zip(tweets, range(len(tweets))):
        #clean tweet
        tweet = re.sub("@[A-Za-z0-9_]+","", tweet)
        tweet = re.sub(r"http\S+", "", tweet)
        tweet = re.sub(r'\n','', tweet)
        tweet = re.sub(r'\r','', tweet)

        #stopword
        tweet = ' '.join([word for word in tweet.split() if word not in cached_stopwords])

        #separate hashtags from tweets
        hashtags = re.findall("#[a-zA-z0-9]+", tweet)
        hashtag_list[i] = hashtags
        tweet = re.sub("#[A-Za-z0-9_]+","", tweet)

        #stemming/lemmatization
        lemmatizer = WordNetLemmatizer()
        tweet = ' '.join([lemmatizer.lemmatize(word) for word in tweet.split()])
        parsed_tweets[i] = tweet

    df.loc[:,'parsed_tweets'] = parsed_tweets
    df.loc[:,'hashtags'] = hashtag_list

    #drop abbreviated tweets
    df = df[~(df['text'].str.contains('…', regex=False))]
    df.to_csv('../data/parsed.csv', index=False)
    return df
