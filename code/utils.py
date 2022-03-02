import pandas as pd
import fasttext
import re
import numpy as np
from googleapiclient import discovery
import json
import keys
import time
from detoxify import Detoxify
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

def get_toxicity(file_name: pd.DataFrame, start_pos: int, end_pos: int) -> pd.DataFrame:
    """
    Run this (with a number of requests) to analyze some amount of tweets for toxicity (via perspective API)

    Parameters
    ----------
    file_name: str
        the file containing the df to be updated
    start_pos: int
        the index of the df to start at
    requests: int
        the index of the df to end at
    
    Returns
    -------
    toxic_df: pd.DataFrame
        dataframe with a new column for toxicity
    """

    #read df
    df = pd.read_csv(file_name)

    #build client
    client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=keys.PERS_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
    )

    #find starting index

    analyze_request = {
    'comment': { 'text': 'friendly greetings from python' },
    'requestedAttributes': {'TOXICITY': {}},
    }


    idx = start_pos
    for i in range(idx, end_pos):
        #don't break the API lol
        time.sleep(1)
        tweet = df['text'].values[i]

        #get rid of mentions
        tweet = re.sub("@[A-Za-z0-9_]+","", tweet)

        analyze_request['comment']['text'] = tweet 
        try:
            response = client.comments().analyze(body=analyze_request).execute()
            tox_score = response['attributeScores']['TOXICITY']['summaryScore']['value']
        except Exception as e:

            print('failed', i, tweet)
            tox_score = -1
        
        df.at[i, 'toxicity'] = tox_score
        if i % 500 == 0:
            df.to_csv(file_name)

    #save df
    df.to_csv(file_name)

    return end_pos

def get_detoxicity(file_name: pd.DataFrame, batch_size: int=10, start_pos: int=0, save_interval: int=10) -> pd.DataFrame:
    """
    Run this (with a number of requests) to analyze some amount of tweets for toxicity (via detoxify)

    Parameters
    ----------
    file_name: str
        the file containing the df to be updated
    batch_size: int
        batch size
    start_pos: int
        the index of the df to start at
    save_interval: int
        how many batches to save after
    
    Returns
    -------
    toxic_df: pd.DataFrame
        dataframe with a new column for toxicity
    """

    #read df
    #add cleaning of links and mentions to this
    df = pd.read_csv(file_name)
    detoxify_model = Detoxify('unbiased')
    for i in range(int((len(df) - start_pos) / batch_size)):
        s = start_pos + batch_size * i
        e = s + batch_size
        predict_update_df(df, detoxify_model, s, e)
        if save_interval > 0 and i % save_interval == 0:
            df.to_csv(file_name) #Note that interrupting during this interval may cause bugs

    
    predict_update_df(df, detoxify_model, int((len(df) - start_pos) / batch_size) + start_pos, len(df))


    df.to_csv(file_name)
    return

def predict_update_df(df, detoxify_model, start_pos, end_pos):
    """
    Helper function for get_toxicity which makes a call to the detoxify model and 
    updates the dataframe in-place

    Parameters
    ----------
    df: pd.DataFrame
        the dataframe to be updated
    detoxify_model: model
        model to use for inference
    start_pos: int
        the index of the df to start at
    end_pos: int
       the index of the df to end at
    
    Returns
    -------
    None
    """
    results = detoxify_model.predict(df['text'][start_pos:end_pos].tolist())
    df.iloc[start_pos:end_pos, df.columns.get_loc('toxicity')] = results['toxicity']
    df.iloc[start_pos:end_pos, df.columns.get_loc('severe_toxicity')] = results['severe_toxicity']
    df.iloc[start_pos:end_pos, df.columns.get_loc('identity_attack')] = results['identity_attack']

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



#if __name__ == '__main__':
    #get_detoxicity('../data/toxicity4.csv', 2, 433770, -1)
