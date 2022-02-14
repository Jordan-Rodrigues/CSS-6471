import pandas as pd
import fasttext
import re
import numpy as np

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


