import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def data_split(df_clean):
    df_train, df_test_val = train_test_split(df_clean, test_size=0.1, random_state=2022, stratify=df_clean["userId"])
    return df_train, df_test_val


def bag_of_words(df_train, df_test):
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(df_train['text'].values.tolist())
    X_test = vectorizer.transform(df_test['text'].values.tolist())
    data_train = np.c_[np.transpose(df_clean["score"].values), X_train.toarray()]
    data_test = np.c_[np.transpose(df_clean["score"].values), X_test.toarray()]
    return data_train, data_test


def word2vec(df_train, df_test):
    return 0


def TFIDF(df_train, df_test):
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", help="vectorisation method, one of: bow, w2v, tfidf",
                        type=int, default=5)
    parser.add_argument("-i", "--input-path", help="input csv file path",
                        type=str, default=5)
    parser.add_argument("-o", "--output-path-train-val", help="output pickle file path for data for modeling",
                        type=str, default=5)
    parser.add_argument("-o", "--output-path-test", help="output pickle file path for data for testing",
                        type=str, default=5)
    args = parser.parse_args()
    return args.method, args.input_path, args.output_path_train_val, args.output_path_test


if __name__ == "__main__":
    method, input_path, output_path_train_val, output_path_test = parse_args()
    df_clean = pd.read_csv(input_path)
    df_train, df_test_val = data_split(df_clean)

    # vectorise
    data_train, data_test = None, None

    # Save results
    with open(output_path_train_val, "rb") as f:
        pickle.dump(data_train, f)

    with open(output_path_test, "rb") as f:
        pickle.dump(data_test, f)
