import pandas as pd
import argparse
from pathlib import Path
from math import sqrt
import contractions
import re
import gzip
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def load_reviews_to_df(input_path):
    reviews_array = []
    dictionary = {}
    with gzip.open(input_path) as raw_data:
        for review in raw_data:
            this_line = review.decode("utf-8").split(":")
            if len(this_line) > 1:
                dictionary[this_line[0]] = this_line[1].strip()
            else:
                reviews_array.append(dictionary)
                dictionary = {}

    col_names = ['productId', 'title', 'price', 'userId',
                'profileName', 'helpfulness', 'score',
                'time', 'summary', 'text']

    reviews = pd.DataFrame(reviews_array)
    reviews.columns = col_names
    reviews[['score']] = reviews[['score']].astype(float)
    reviews['time'] = pd.to_datetime(reviews['time'], unit='s')
    reviews["helpfulness_num"] = reviews["helpfulness"].apply(lambda x: int(x.split("/")[0]))
    reviews["helpfulness_den"] = reviews["helpfulness"].apply(lambda x: int(x.split("/")[1]))
    return reviews


def select_rows(df, min_opinions):
    df["year"] = df['time'].astype('datetime64[ns]').dt.year
    df = df.loc[(df["userId"] != "unknown") & df["year"].isin([2011, 2012, 2013]), ]
    user_no_opinions = df[["userId", "productId"]].groupby("userId").count()
    users = user_no_opinions.loc[user_no_opinions["productId"] >= min_opinions, ].index.values
    return df.loc[df["userId"].isin(users), ["productId", "userId", "score", "text", "helpfulness_num", "helpfulness_den"]]


def confidence(ups, n):
    if n == 0:
        return 0
    z = 1.281551565545
    p = float(ups) / n
    left = p + 1 / (2 * n) * z ** 2
    right = z * sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    under = 1 + 1 / n * z ** 2
    return (left - right) / under


def clean_text(text, wnl):
    text = str(text)
    text = contractions.fix(text, slang=True)
    text = text.lower()
    text = re.sub(r"\d+", "", re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text))
    words = word_tokenize(text)
    return " ".join([wnl.lemmatize(i) for i in words])


def prepare_data(min_opinions, input_path, output_path):
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    wnl = WordNetLemmatizer()
    df = load_reviews_to_df(input_path)
    df_cleaned = select_rows(df, min_opinions)
    df_cleaned["wilson_score"] = df_cleaned.apply(
        lambda row: confidence(row["helpfulness_num"], row["helpfulness_den"]), axis=1)
    df_cleaned["text"] = df_cleaned["text"].apply(clean_text, wnl=wnl)
    df_cleaned.to_csv(output_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--min-opinions", help="minimal number of opinions required to include user in data",
                        type=int, default=5)
    parser.add_argument("-i", "--input-path", help="input gz file path",
                        type=str, default=5)
    parser.add_argument("-o", "--output-path", help="output csv file path",
                        type=str, default=5)
    args = parser.parse_args()
    return args.min_opinions, args.input_path, args.output_path


if __name__ == "__main__":
    min_opinions, input_path, output_path = parse_args()
    prepare_data(min_opinions, input_path, output_path)
