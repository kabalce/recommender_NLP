import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", help="na imputation method, one of: ...",
                        type=int, default=5)
    parser.add_argument("-i", "--input-path-df", help="input pickle file path df",
                        type=str, default=5)
    parser.add_argument("-i", "--input-path-pkl", help="input pickle file path pkl",
                        type=str, default=5)
    parser.add_argument("-o", "--output-path-df", help="output pickle file path df",
                        type=str, default=5)
    parser.add_argument("-p", "--output-path-pkl", help="output pickle file path pkl",
                        type=str, default=5)
    args = parser.parse_args()
    return args.method, args.input_path_train, args.input_path_test, args.output_path_train, args.output_path_test


if __name__ == "__main__":
    method, input_path_train, input_path_test, output_path_train, output_path_test = parse_args()
    with open(input_path_train, "rb") as f:
        data_train = pickle.load(f)
    with open(input_path_test, "rb") as f:
        data_test = pickle.load(f)
