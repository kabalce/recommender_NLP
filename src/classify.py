import pickle
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import argparse
import json


def xgboost(X_train, y_train, hyperparameters):
    model = XGBClassifier(**hyperparameters)
    model.fit(X_train, y_train)
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", help="vectorisation method, one of: xgb, ...",
                        type=int)
    parser.add_argument("-i", "--input-path-train", help="output pickle file path for data for modeling",
                        type=str)
    parser.add_argument("-j", "--input-path-test", help="output pickle file path for data for testing",
                        type=str)
    parser.add_argument("-k", "--input-path-nlp", help="output pickle file path for data for nlp training",
                        type=str)
    parser.add_argument("-o", "--output-path-train", help="output pickle file path for data for modeling",
                        type=str)
    parser.add_argument("-u", "--output-path-test", help="output pickle file path for data for testing",
                        type=str)
    parser.add_argument("-n", "--output-path-nlp", help="output pickle file path for data for nlp training",
                        type=str)
    parser.add_argument("-p", "--params", help="hyperparameters in json format",
                        type=str, default='{objective="multi:softmax", learning_rate=0.3}')
    args = parser.parse_args()
    return args.method, args.input_path_nlp, args.input_path_train, args.input_path_test, \
           args.output_path_nlp, args.output_path_train, args.output_path_test, args.params


if __name__ == "__main__":
    method, input_path_nlp, input_path_train, input_path_test, output_path_nlp, output_path_train, output_path_test, params = parse_args()
    hyperparameters = json.loads(params)
    with open(input_path_nlp, "rb") as f:
        data_nlp = pickle.load(f)
    with open(input_path_train, "rb") as f:
        data_train = pickle.load(f)
    with open(input_path_test, "rb") as f:
        data_test = pickle.load(f)
    X_nlp_train, y_nlp_train = data_nlp[:, 1:], data_nlp[:, 0]

    method_functions = {"xgb": xgboost}

    # make classification
    assert method in method_functions.keys(), f"Unrecognised method: {method}"
    model = method_functions[method](X_nlp_train, y_nlp_train, hyperparameters)

    # build predictions
    results_train = 0
    with open(output_path_train, "wb") as f:
        pickle.dump(results_train, f)

    if data_test is not None and output_path_test is not None:
        X_test, y_test = data_test[:, 1:], data_test[:, 0]
        results_test = 0
        with open(output_path_test, "wb") as f:
            pickle.dump(results_test, f)
