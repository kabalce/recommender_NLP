import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec


def data_split(df_clean):
    df_nlp = df_clean.loc[df_clean["year"].isin([2010, 2011]), ]
    df_train, df_test_val = train_test_split(df_clean.loc[~df_clean["year"].isin([2010, 2011]), ], test_size=0.2, random_state=2022,
                                             stratify=df_clean.loc[~df_clean["year"].isin([2010, 2011]), "userId"])
    return df_nlp, df_train, df_test_val


def bag_of_words(df_nlp, df_train, df_test):
    vectorizer = CountVectorizer()
    X_nlp = vectorizer.fit_transform(df_nlp['text'].values.tolist())
    X_train = vectorizer.transform(df_train['text'].values.tolist())
    X_test = vectorizer.transform(df_test['text'].values.tolist())
    data_nlp = np.c_[np.transpose(df_nlp["score"].values), X_nlp.toarray()]
    data_train = np.c_[np.transpose(df_train["score"].values), X_train.toarray()]
    data_test = np.c_[np.transpose(df_test["score"].values), X_test.toarray()]
    return data_nlp, data_train, data_test


def word2vec(df_nlp, df_train, df_test):
    sent_train = [str(row).split() for row in df_nlp['text']]
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
            data_nlp = words_mean
        else:
            data_nlp = np.concatenate((data_nlp, words_mean))

    data_nlp = np.c_[np.transpose(df_nlp["score"].values), data_nlp]

    # train
    sent_test = [row.split() for row in df_train['text']]
    for i in range(len(sent_test)):
        word_tokens = sent_test[i]
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

    return data_nlp, data_train, data_test


def TFIDF(df_nlp, df_train, df_test):
    vectorizer = TfidfVectorizer()
    X_nlp = vectorizer.fit_transform(df_nlp['text'].values.tolist())
    X_train = vectorizer.transform(df_train['text'].values.tolist())
    X_test = vectorizer.transform(df_test['text'].values.tolist())
    data_nlp = np.c_[np.transpose(df_nlp["score"].values), X_nlp.toarray()]
    data_train = np.c_[np.transpose(df_train["score"].values), X_train.toarray()]
    data_test = np.c_[np.transpose(df_test["score"].values), X_test.toarray()]
    return data_nlp, data_train, data_test


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", help="vectorisation method, one of: bow, w2v, tfidf",
                        type=str)
    parser.add_argument("-i", "--input-path", help="input csv file path",
                        type=str)
    parser.add_argument("-o", "--output-path-train", help="output pickle file path for data for modeling",
                        type=str)
    parser.add_argument("-u", "--output-path-test", help="output pickle file path for data for testing",
                        type=str)
    parser.add_argument("-n", "--output-path-nlp", help="output pickle file path for data for nlp training",
                        type=str)
    args = parser.parse_args()
    return args.method, args.input_path, args.output_path_train, args.output_path_test,args.output_path_nlp


if __name__ == "__main__":
    method, input_path, output_path_train, output_path_test, output_path_nlp = parse_args()
    df_clean = pd.read_csv(input_path)
    df_nlp, df_train, df_test = data_split(df_clean)

    method_functions = {"bow": bag_of_words,
                        "w2v": word2vec,
                        "tfidf": TFIDF}

    # vectorise
    assert method in method_functions.keys(), f"Unrecognised method: {method}"
    data_nlp, data_train, data_test = method_functions[method](df_nlp, df_train, df_test)

    # Save results
    with open(output_path_train, "wb") as f:
        pickle.dump(data_train, f)

    with open(output_path_test, "wb") as f:
        pickle.dump(data_test, f)

    with open(output_path_nlp, "wb") as f:
        pickle.dump(data_nlp, f)
