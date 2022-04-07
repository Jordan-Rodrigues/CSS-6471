import pandas as pd
import re
import time

from detoxify import Detoxify

# DEPRECATED
# from googleapiclient import discovery

# DEPRECATED
'''def get_toxicity(file_name: pd.DataFrame, start_pos: int, end_pos: int) -> pd.DataFrame:
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

    return end_pos'''


def get_toxicity_detoxify(
    df: pd.DataFrame,
    start_pos: int = 0,
    save_interval: int = 1000,
    exists: bool = False,
    file_name: str = "../data/toxicity.csv",
) -> pd.DataFrame:
    """
    Run this to analyze some amount of tweets for toxicity (via detoxify)

    Parameters
    ----------
    df: pd.DataFrame
        dataframe to analyze toxicity of
    start_pos: int
        the index of the df to start at
    save_interval: int
        how many tweets to save after
    exists: bool
        flag on whether to generate toxicity or read from file
    file_name: str
        file name to save or read

    Returns
    -------
    toxic_df: pd.DataFrame
        dataframe with a new column for toxicity
    """
    if exists:
        return pd.read_csv(file_name)

    if "toxicity" not in df.columns:
        df["toxicity"] = -1.0
    if "severe_toxicity" not in df.columns:
        df["severe_toxicity"] = -1.0
    if "identity_attack" not in df.columns:
        df["identity_attack"] = -1.0

    detoxify_model = Detoxify("unbiased")
    print("loaded detoxify model")
    print("do not terminate when saving... is shown, only after saved")

    tweets = df["text"]
    for tweet, i in zip(tweets, range(len(tweets))):
        # clean tweet
        tweet = re.sub(r"\n", "", tweet)
        tweet = re.sub(r"\r", "", tweet)
        tweet = re.sub("@[A-Za-z0-9_]+", "", tweet)
        tweet = re.sub(r"http\S+", "", tweet)

        results = detoxify_model.predict(tweet)
        df.iloc[start_pos + i, df.columns.get_loc("toxicity")] = results["toxicity"]
        df.iloc[start_pos + i, df.columns.get_loc("severe_toxicity")] = results[
            "severe_toxicity"
        ]
        df.iloc[start_pos + i, df.columns.get_loc("identity_attack")] = results[
            "identity_attack"
        ]

        if save_interval > 0 and i % save_interval == 0:
            print("saving...", end="\r")
            df.to_csv(file_name, index=False)
            print("saved", i)

    # save df
    df.to_csv(file_name, index=False)

    print("saved all", len(tweets))


def get_toxicity_detoxify_batched(
    file_name: str, batch_size: int = 2, start_pos: int = 0, save_interval: int = 10
) -> pd.DataFrame:
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

    # read df
    # add cleaning of links and mentions to this
    df = pd.read_csv(file_name)

    if "toxicity" not in df.columns:
        df["toxicity"] = -1.0
    if "severe_toxicity" not in df.columns:
        df["severe_toxicity"] = -1.0
    if "identity_attack" not in df.columns:
        df["identity_attack"] = -1.0

    detoxify_model = Detoxify("unbiased")
    print("loaded detoxify model")
    print("do not terminate when saving... is shown, only after saved")

    s = start_pos
    e = s + batch_size

    tweets = df["text"]
    for tweet, i in zip(tweets, range(len(tweets))):
        # clean tweet
        tweet = re.sub(r"\n", "", tweet)
        tweet = re.sub(r"\r", "", tweet)
        tweet = re.sub("@[A-Za-z0-9_]+", "", tweet)
        tweet = re.sub(r"http\S+", "", tweet)

    for i in range(int((len(df) - start_pos) / batch_size)):
        s += batch_size
        e += batch_size
        predict_update_df(df, detoxify_model, s, e)
        if save_interval > 0 and i % save_interval == 0:
            print("saving...", end="\r")
            df.to_csv(file_name, index=False)
            print("saved", s, "+", batch_size)

    predict_update_df(df, detoxify_model, e, len(df))

    # save df
    df.to_csv(file_name, index=False)

    print("saved all", e)


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
    results = detoxify_model.predict(df["text"][start_pos:end_pos].tolist())
    df.iloc[start_pos:end_pos, df.columns.get_loc("toxicity")] = results["toxicity"]
    df.iloc[start_pos:end_pos, df.columns.get_loc("severe_toxicity")] = results[
        "severe_toxicity"
    ]
    df.iloc[start_pos:end_pos, df.columns.get_loc("identity_attack")] = results[
        "identity_attack"
    ]


if __name__ == "__main__":
    get_toxicity_detoxify(
        pd.read_csv("../data/toxicity.csv"), start_pos=0, save_interval=2000
    )
