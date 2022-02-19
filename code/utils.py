import pandas as pd
import fasttext
import re
import numpy as np
from googleapiclient import discovery
import json
import keys
import time
from detoxify import Detoxify

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
    

    Returns
    -------
    english_df
        df, but only with english rows
    exists : bool, optional
        Flag on whether to generate eng or read from file
    """
    if exists:
        return pd.read_csv('../data/english.csv')

    PRETRAINED_MODEL_PATH = '../data/lid.176.bin'
    model = fasttext.load_model(PRETRAINED_MODEL_PATH)
    text_arr = df['text'].values
    mask = []

    for text in text_arr: 
        #remove mentions
        text = re.sub("@[A-Za-z0-9_]+","", text)
        #remove newlines
        text = re.sub(r'\n','', text)
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
    Run this (with a number of requests) to analyze some amount of tweets for toxicity

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
            print(e)
            print('failed', i, tweet)
            tox_score = -1
        
        df.at[i, 'toxicity'] = tox_score
        if i % 500 == 0:
            df.to_csv(file_name, index=False)

    #save df
    df.to_csv(file_name, index=False)

    return end_pos

def get_detoxicity(file_name: pd.DataFrame, batch_size: int=10, start_pos: int=0, save_interval: int=10) -> pd.DataFrame:
    """
    Run this (with a number of requests) to analyze some amount of tweets for toxicity

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
    df = pd.read_csv(file_name)
    print('loaded df')
    detoxify_model = Detoxify('unbiased')
    print('loaded detoxify model')
    print('do not terminate when saving... is shown, only after saved')

    for i in range(int((len(df) - start_pos) / batch_size)):
        s = start_pos + batch_size * i
        e = s + batch_size
        predict_update_df(df, detoxify_model, s, e)
        if save_interval > 0 and i % save_interval == 0:
            print('saving...', end='\r')
            df.to_csv(file_name, index=False)
            print('saved', start_pos + batch_size * i) 

    
    predict_update_df(df, detoxify_model, int((len(df) - start_pos) / batch_size) + start_pos, len(df))

    #save df
    df.to_csv(file_name, index=False)

    print('saved')

    return

def predict_update_df(df, detoxify_model, start_pos, end_pos):
    results = detoxify_model.predict(df['text'][start_pos:end_pos].tolist())

    df.iloc[start_pos:end_pos, df.columns.get_loc('toxicity')] = results['toxicity']
    df.iloc[start_pos:end_pos, df.columns.get_loc('severe_toxicity')] = results['severe_toxicity']
    df.iloc[start_pos:end_pos, df.columns.get_loc('identity_attack')] = results['identity_attack']


if __name__ == '__main__':
    # get_detoxicity(file_name: pd.DataFrame, batch_size: int=10, start_pos: int=0, save_interval: int=10)
    get_detoxicity('../data/toxicity4.csv', 2, 433770, -1)