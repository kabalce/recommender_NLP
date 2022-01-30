import pickle
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import argparse


def xgboost(X_train, y_train):
    model = XGBClassifier(objective="multi:softmax", learning_rate=0.3)  # TODO: use tuned hyperparams
    model.fit(X_train, y_train)
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", help="vectorisation method, one of: xgb, ...",
                        type=int)
    parser.add_argument("-i", "--input-path-train", help="input pickle file path train",
                        type=str)
    parser.add_argument("-j", "--input-path-test", help="input pickle file path test",
                        type=str, default=None)
    parser.add_argument("-o", "--output-path-train", help="output pickle file path train",
                        type=str)
    parser.add_argument("-u", "--output-path-test", help="output pickle file path test",
                        type=str, default=None)
    parser.add_argument("-p", "--params-path", help="output pickle file path test",
                        type=str)
    args = parser.parse_args()
    return args.method, args.input_path_train, args.input_path_test, args.output_path_train, args.output_path_test, args.params_path


if __name__ == "__main__":
    method, input_path_train, input_path_test, output_path_train, output_path_test, params_path = parse_args()
    with open(input_path_train, "rb") as f:
        data_train = pickle.load(f)
    with open(input_path_test, "rb") as f:
        data_test = pickle.load(f)
    X_train, y_train = data_train[:, 1:], data_train[:, 0]

    method_functions = {"xgb": xgboost}

    # make classification
    assert method in method_functions.keys(), f"Unrecognised method: {method}"
    model = method_functions[method]

    # build predictions
    results_train = 0
    with open(output_path_train, "wb") as f:
        pickle.dump(results_train, f)

    if data_test != None and output_path_test != None:
        X_test, y_test = data_test[:, 1:], data_test[:, 0]
        results_test = 0
        with open(output_path_test, "wb") as f:
            pickle.dump(results_test, f)
