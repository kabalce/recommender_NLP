import pandas as pd
import argparse
import os
from pathlib import Path
from math import sqrt


def clean_data(df, min_opinions):
    df["year"] = df['time'].astype('datetime64[ns]').dt.year
    df = df.loc[(df["userId"] != "unknown") & df["year"].isin([2011, 2012, 2013]), ]
    user_no_opinions = df[["userId", "productId"]].groupby("userId").count()
    users = user_no_opinions.loc[user_no_opinions["productId"] >= min_opinions, ].index.values
    return df.loc[df["userId"].isin(users), ]


def confidence(ups, n):
    if n == 0:
        return 0
    else:
        z = 1.281551565545
        p = float(ups) / n
        left = p + 1 / (2 * n) * z ** 2
        right = z * sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
        under = 1 + 1 / n * z ** 2
        return (left - right) / under


def preprare_data(min_opinions):
    source_path = "data/raw"
    nlp_path = "data/nlp"
    recommend_path = "data/recommend"
    Path(nlp_path).mkdir(exist_ok=True, parents=True)
    Path(recommend_path).mkdir(exist_ok=True, parents=True)
    inputs = [f for f in os.listdir(source_path) if not f.startswith('.')]
    for file_name in inputs:
        print(f"Processing {file_name}")
        df = pd.read_csv(f"{source_path}/{file_name}")
        print(f"\tOriginal size:       {df.shape}")
        df_cleaned = clean_data(df, min_opinions)
        print(f"\tSize after cleaning: {df_cleaned.shape}")
        df_cleaned["wilson_score"] = df_cleaned.apply(lambda row: confidence(row["helpfulness_num"], row["helpfulness_den"]), axis=1)
        df_cleaned[["productId", "userId", "score", "wilson_score", "helpfulness_num", "helpfulness_den"]].to_csv(f"{recommend_path}/{file_name}")
        df_cleaned[["text", "summary", "score", "helpfulness_num", "helpfulness_den"]].to_csv(f"{nlp_path}/{file_name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--min-opinions", help="minimal number of opinions required to include user in data",
                        type=int, default=5)
    args = parser.parse_args()
    return args.min_opinions

if __name__ == "__main__":
    min_opinions = parse_args()
    preprare_data(min_opinions)