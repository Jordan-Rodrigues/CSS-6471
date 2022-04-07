from scipy import stats
import numpy as np
import consts
import pandas as pd


class StatsDF:
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
        self.full = df
        self.tokyo = df[df["date"].str.slice(0, 4) == "2021"]
        self.beijing = df[df["date"].str.slice(0, 4) == "2022"]

    def KS_test(self, column1, column2, df_type, alternative):
        """
        Runs a KS test on two continuous values

        Parameters
        ----------
        column1: str
            the first column to compare
        column2: str
            the second column to compare
        df_type: str
            what type of comparison to run (self, across events, etc.)
        alternative: str
            what alternative hypothesis
        Returns
        -------
        stat: float
            KS test statistic
        pvalue: float
            probability null holds true
        """
        if df_type == "both":
            col_1 = self.full[column1].values
            col_2 = self.full[column2].values
        else:
            col_1 = self.tokyo[column1].values
            col_2 = self.beijing[column2].values
        stat, pvalue = stats.kstest(col_2, col_1)
        return stat, pvalue

    def chisquare(self, column1, column2, df_type, alternative):
        """
        Runs a Chi Square test on two discrete values

        Parameters
        ----------
        column1: str
            the first column to compare
        column2: str
            the second column to compare
        df_type: str
            what type of comparison to run (self, across events, etc.)
        alternative: str
            what alternative hypothesis
        Returns
        -------
        chisq: float
            CS test statistic
        pvalue: float
            probability null holds true
        """
        if df_type == "full":
            col_1 = self.full[column1].value_counts().values
            col_2 = self.full[column2].value_counts()
        elif df_type == "year":
            col_1 = self.tokyo[column1].value_counts().values
            col_2 = self.beijing[column2].value_counts().values
        else:
            col_1 = self.iso[column1].value_counts().values
            col_2 = self.pop[column2].value_counts().values

        # set bigger to be consistent
        if np.sum(col_2) > np.sum(col_1):
            col_1, col_2 = col_2, col_1

        # normalize
        col_1 = [x / np.sum(col_1) for x in col_1]
        col_1 = [x * np.sum(col_2) for x in col_1]

        chisq, pvalue = stats.chisquare(col_2, col_1)
        return chisq, pvalue

    def split_popularity(self, pf_fo, pf_fr, pc_fo, pc_fr):
        """
        Calculates mean toxicity/sentiment at various levels of user popularity

        Parameters
        ----------
        pf_fo: int
            percentile floor for followers
        pf_fr: int
            percentile floor for friends
        pc_fo: int
            percentile ceiling for followers
        pc_fr: int
            percentile ceiling for friends
        Returns
        -------
        tox_mean: float
            average tox value for bucket
        sent_mean: float
            average sent value for bucket
        tox_values: np.ndarray
            list of tox values in bucket
        sent_values: np.ndarray
            list of sent values in bucket
        """

        df = self.full

        # get score cutoffs
        fos_f = stats.scoreatpercentile(df["user_followers"], pf_fo)
        frs_f = stats.scoreatpercentile(df["user_friends"], pf_fr)

        fos_c = stats.scoreatpercentile(df["user_followers"], pc_fo)
        frs_c = stats.scoreatpercentile(df["user_friends"], pc_fr)

        # create and stratify with score buckets
        df["isPopular"] = False
        df.loc[
            ((df["user_followers"] >= fos_f) & (df["user_followers"] < fos_c))
            & ((df["user_friends"] >= frs_f) & (df["user_friends"] < frs_c)),
            "isPopular",
        ] = True
        self.iso = df[df["isPopular"] == False]
        self.pop = df[df["isPopular"] == True]

        # map words to ints and calculate mean
        tox_mean = np.mean(self.pop["tox_binned"].map(consts.TOX_MAP))
        sent_mean = np.mean(self.pop["sent_binned"].map(consts.SENT_MAP).astype(int))

        # also return values for stat testing
        tox_values = self.pop["tox_binned"].map(consts.TOX_MAP).values
        sent_values = self.pop["sent_binned"].map(consts.SENT_MAP).values

        return tox_mean, sent_mean, tox_values, sent_values

    def country_analysis(self, sample_cutoff):
        """
        Calculates mean toxicity/sentiment for various nations

        Parameters
        ----------
        sample_cutoff: int
            used to determine how many nations to include
        Returns
        -------
        new_df: pd.DataFrame
            pandas dataframe with country column, tox value column, and sent value column (not melted)
        """
        countries = self.full[self.full["clean_loc"] != "None"]
        countries = countries["clean_loc"].value_counts()
        countries = countries[countries > sample_cutoff]
        countries = countries.index.values
        temp_df = self.full[self.full["clean_loc"].isin(countries)]
        tox_list = []
        sent_list = []

        new_df = pd.DataFrame()
        new_df["Country"], new_df["Toxicity"], new_df["Sentiment"] = None, 0, 0

        for country in countries:
            tox_slice = temp_df[temp_df["clean_loc"] == country]["tox_binned"].map(
                consts.TOX_MAP
            )
            sent_slice = (
                temp_df[temp_df["clean_loc"] == country]["sent_binned"]
                .map(consts.SENT_MAP)
                .astype(int)
            )
            country_slice = [country] * len(sent_slice)
            df_slice = pd.DataFrame(
                {
                    "Country": country_slice,
                    "Toxicity": tox_slice,
                    "Sentiment": sent_slice,
                }
            )
            new_df = pd.concat([df_slice, new_df])
        return new_df

    def topic_analysis(self, mode, topic_1, topic_2, column):
        """
        Runs a Chi Square test on two discrete values.

        Parameters
        ----------
        mode: str
            whether the topics are both in one event (and which event), or not
        topic_1: str
            topic_name of the first topic (assumes 2020 when mode != 2020/2022)
        topic_2: str
            topic_name of the second topic (assumes 2022 when mode != 2020/2022)
        column: str
            which column to compare
        Returns
        -------
        chisq: float
            CS test statistic
        pvalue: float
            probability null holds true
        """
        if mode == "2020":
            slice_1 = self.tokyo[self.tokyo["topic_name"] == topic_1][column]
            slice_2 = self.tokyo[self.tokyo["topic_name"] == topic_2][column]
        elif mode == "2022":
            slice_1 = self.beijing[self.beijing["topic_name"] == topic_1][column]
            slice_2 = self.beijing[self.beijing["topic_name"] == topic_2][column]
        else:
            slice_1 = self.tokyo[self.tokyo["topic_name"] == topic_1][column]
            slice_2 = self.beijing[self.beijing["topic_name"] == topic_2][column]

        col_1 = slice_1.value_counts().values
        col_2 = slice_2.value_counts().values

        # set bigger to be consistent
        if np.sum(col_2) > np.sum(col_1):
            col_1, col_2 = col_2, col_1

        # normalize
        col_1 = [x / np.sum(col_1) for x in col_1]
        col_1 = [x * np.sum(col_2) for x in col_1]

        chisq, pvalue = stats.chisquare(col_2, col_1)
        return chisq, pvalue
