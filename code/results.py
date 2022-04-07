import pandas as pd
import stats
import scipy.stats as sstat
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


def run_data():
    print(
        "Reading data from csv... to build data from source, view pull_tweets/collectData.py. This script cannot be run for API Security Reasons"
    )
    df = pd.read_csv("../final_data/topic.csv")
    print("Numbers of Rows:", len(df), "\n")
    print("Columns", [x for x in df.columns], "\n")
    print("Sample", df.head(), "\n")


def run_stats():
    df = pd.read_csv("../final_data/topic.csv")

    # add year variable for plotting
    df["Year"] = df["date"].str.slice(0, 4)

    # instantiate stats class
    stat_df = stats.StatsDF(df)

    # Tox Between Events
    chisq, pvalue = stat_df.chisquare("tox_binned", "tox_binned", "year", "two-sided")
    print(
        "Two-Sided Chi Square Test Statistic for Tweet Toxicity in Tokyo vs Beijing",
        "\n",
    )
    print("chisq", chisq, "pvalue", pvalue)
    if pvalue < 0.01:
        print("The difference in toxicity is statistically significant\n")
    else:
        print("The difference in toxicity is not statistically significant\n")

    # Sent Between Events
    print(
        "Two-Sided Chi Square Test Statistic for Tweet Sentiment in Tokyo vs Beijing",
        "\n",
    )
    chisq, pvalue = stat_df.chisquare("sent_binned", "sent_binned", "year", "two-sided")
    print("chisq", chisq, "pvalue", pvalue)
    if pvalue < 0.01:
        print("The difference in sentiment is statistically significant")
    else:
        print("The difference in sentiment is not statistically significant")

    # Tox/Sent Between Popularity Levels
    low, step, high = 0, 20, 101
    tox_scores, sent_scores, tox_val_list, sent_val_list = [], [], [], []

    for i in range(low, high - step, step):
        tox_mean, sent_mean, tox_vals, sent_vals = stat_df.split_popularity(
            i, i, i + step, i + step
        )
        tox_scores.append(tox_mean)
        sent_scores.append(sent_mean)
        tox_val_list.append(tox_vals)
        sent_val_list.append(sent_vals)
    print("\n")
    print("Toxicity Score for 0 - 20th Popularity Percentile")
    print(tox_scores[0])

    print("Toxicity Score for 20 - 40th Popularity Percentile")
    print(tox_scores[1])

    print("Toxicity Score for 40 - 60th Popularity Percentile")
    print(tox_scores[2])

    print("Toxicity Score for 60 - 80th Popularity Percentile")
    print(tox_scores[3])

    print("Toxicity Score for 80 - 100th Popularity Percentile")
    print(tox_scores[4])
    print("\n")

    k_stat, p_value = sstat.kruskal(
        tox_val_list[0],
        tox_val_list[1],
        tox_val_list[2],
        tox_val_list[3],
        tox_val_list[4],
    )

    print("Kruskal Test Statistic for Toxicity Amongst Popularity Groups")
    print("Test Stat", k_stat, "pvalue", p_value)
    if p_value < 0.01:
        print(
            "The toxicity difference across the popularity groups is statistically significant"
        )
    else:
        print(
            "The toxicity difference across the popularity groups is not statistically significant"
        )

    print("\n")
    print("Sentiment Score for 0 - 20th Popularity Percentile")
    print(sent_scores[0])

    print("Sentiment Score for 20 - 40th Popularity Percentile")
    print(sent_scores[1])

    print("Sentiment Score for 40 - 60th Popularity Percentile")
    print(sent_scores[2])

    print("Sentiment Score for 60 - 80th Popularity Percentile")
    print(sent_scores[3])

    print("Sentiment Score for 80 - 100th Popularity Percentile")
    print(sent_scores[4])
    print("\n")

    k_stat, p_value = sstat.kruskal(
        sent_val_list[0],
        sent_val_list[1],
        sent_val_list[2],
        sent_val_list[3],
        sent_val_list[4],
    )

    print("Kruskal Test Statistic for Sentiment Amongst Popularity Groups")
    print("Test Stat", k_stat, "pvalue", p_value)
    if p_value < 0.01:
        print(
            "The sentiment difference across the popularity groups is statistically significant"
        )
    else:
        print(
            "The sentiment difference across the popularity groups is not statistically significant"
        )

    # Country Based Analysis
    print("\nVisualizing Differences in Toxicity/Sentiment by Country in Plot\n")
    new_df = stat_df.country_analysis(1000)
    matplotlib.style.use("ggplot")
    fig = plt.figure(figsize=(10, 10))  # Create matplotlib figure

    ax = fig.add_subplot(111)  # Create matplotlib axes
    ax2 = ax.twinx()  # Create another axes that shares the same x-axis as ax.

    width = 0.3

    # calculator error
    err_df = new_df.groupby(["Country"]).describe()
    err_df.columns
    err_df[("Toxicity", "se")] = err_df[("Toxicity", "std")] / np.sqrt(
        err_df[("Toxicity", "count")]
    )
    err_df[("Sentiment", "se")] = err_df[("Sentiment", "std")] / np.sqrt(
        err_df[("Sentiment", "count")]
    )

    # plot
    pivot_df = (
        pd.pivot_table(
            index="Country",
            values=["Toxicity", "Sentiment"],
            aggfunc=np.mean,
            data=new_df,
        )
        .sort_values(by="Toxicity", ascending=False)["Toxicity"]
        .plot(
            kind="bar",
            width=width,
            ax=ax,
            position=1,
            color="red",
            legend=False,
            rot=50,
            alpha=0.5,
            yerr=err_df[("Toxicity", "se")],
        )
    )

    pivot_df = (
        pd.pivot_table(
            index="Country",
            values=["Sentiment", "Toxicity"],
            aggfunc=np.mean,
            data=new_df,
        )
        .sort_values(by="Toxicity", ascending=False)["Sentiment"]
        .plot(
            kind="bar",
            width=width,
            ax=ax2,
            position=0,
            color="blue",
            legend=False,
            alpha=0.5,
            yerr=err_df[("Sentiment", "se")],
        )
    )

    ax.set_ylabel("Toxicity")
    ax2.set_ylabel("Sentiment")
    ax2.set_title("Text Characteristics by Nation")
    ax.set_xlabel("")

    fig.legend()

    plt.show()


def run_topics():
    topic_df = pd.read_csv("../final_data/topic.csv")
    ts_df = stats.StatsDF(topic_df)

    print("Testing Difference in Toxicity in 2022 Topics (Kamila vs FrozenFin)")
    chsq, p_value = ts_df.topic_analysis(
        "2022", "KamilaValieva", "FrozenFin", "tox_binned"
    )
    print("Test Stat", chsq, "PValue", p_value)
    if p_value < 0.01:
        print(
            "The difference in toxicity between KamilaValieva discussion and FrozenFin discussion is statistically significant\n"
        )
    else:
        print(
            "The difference in toxicity between KamilaValieva discussion and FrozenFin discussion is not statistically significant\n"
        )

    print("Testing Difference in Sentiment in 2022 Topics (Kamila vs FrozenFin)")
    chsq, p_value = ts_df.topic_analysis(
        "2022", "KamilaValieva", "FrozenFin", "sent_binned"
    )
    print("Test Stat", chsq, "PValue", p_value)
    if p_value < 0.01:
        print(
            "The difference in sentiment between KamilaValieva discussion and FrozenFin discussion is statistically significant"
        )
    else:
        print(
            "The difference in sentiment between KamilaValieva discussion and FrozenFin discussion is not statistically significant"
        )


if __name__ == "__main__":
    run_data()
    run_stats()
    run_topics()
