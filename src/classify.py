import pickle
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import argparse


def xgboost(X_train, y_train, X_test):
    model = XGBClassifier(objective="multi:softmax", learning_rate=0.3)  # TODO: use tuned hyperparams
    model.fit(X, Y)
    y_pred = model.predict(X)
    print(f"Accuracy on train sample: {accuracy_score(y_pred, y)}")
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", help="vectorisation method, one of: bow, w2v, tfidf",
                        type=int, default=5)
    parser.add_argument("-i", "--input-path-train", help="input pickle file path train",
                        type=str, default=5)
    parser.add_argument("-i", "--input-path-test", help="input pickle file path test",
                        type=str, default=5)
    parser.add_argument("-o", "--output-path-train", help="output pickle file path train",
                        type=str, default=5)
    parser.add_argument("-o", "--output-path-test", help="output pickle file path test",
                        type=str, default=5)
    args = parser.parse_args()
    return args.method, args.input_path_train, args.input_path_test, args.output_path_train, args.output_path_test


if __name__ == "__main__":
    method, input_path_train, input_path_test, output_path_train, output_path_test = parse_args()
    with open(input_path_train, "rb") as f:
        data_train = pickle.load(f)
    with open(input_path_test, "rb") as f:
        data_test = pickle.load(f)
    X_train, y_train = data_train[:, 1:], data_train[:, 0]
    X_test, y_test = data_test[:, 1:], data_test[:, 0]

    # make classification
    model = None
    results_train = 0
    results_test = 0

    with open(output_path_train, "rb") as f:
        pickle.dump(results_train, f)
    with open(output_path_test, "rb") as f:
        pickle.dump(results_test, f)
