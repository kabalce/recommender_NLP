import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec


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
    sent_train = [row.split() for row in df_train['text']]
    phrases = Phrases(sent_train, min_count=30, progress_per=10000)
    bigram = Phraser(phrases)
    sentences = bigram[sent_train]

    w2v_model = Word2Vec(sentences=sentences, vector_size=300)
    w2v_model.build_vocab(sentences, progress_per=10000)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

    for i in range(len(sent_train)):
        word_tokens = sent_train[i]
        words_mean = np.mean(w2v_model.wv.vectors_for_all(word_tokens).vectors, axis=0).reshape(-1, 300)
        if i == 0:
            data_train = words_mean
        else:
            data_train = np.concatenate((data_train, words_mean))

    data_train = np.c_[np.transpose(df_train["score"].values), data_train]

    # test
    sent_test = [row.split() for row in df_test['text']]
    for i in range(len(sent_test)):
        word_tokens = sent_test[i]
        words_mean = np.mean(w2v_model.wv.vectors_for_all(word_tokens).vectors, axis=0).reshape(-1, 300)
        if i == 0:
            data_test = words_mean
        else:
            data_test = np.concatenate((data_test, words_mean))

    data_test = np.c_[np.transpose(df_test["score"].values), data_test]

    return data_train, data_test


def TFIDF(df_train, df_test):
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(df_train['text'].values.tolist())
    X_test = vectorizer.transform(df_test['text'].values.tolist())
    data_train = np.c_[np.transpose(df_clean["score"].values), X_train.toarray()]
    data_test = np.c_[np.transpose(df_clean["score"].values), X_test.toarray()]
    return data_train, data_test


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
