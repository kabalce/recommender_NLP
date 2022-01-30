import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", help="vectorisation method, one of: bow, w2v, tfidf",
                        type=int)
    parser.add_argument("-i", "--input-path-train", help="input pickle file path train",
                        type=str)
    parser.add_argument("-o", "--output-path-train", help="output pickle file path train",
                        type=str)
    args = parser.parse_args()
    return args.method, args.input_path_train, args.output_path_train


if __name__ == "__main__":
    method, input_path_train, output_path_train = parse_args()

    with open(input_path_train, "rb") as f:
        data_train = pickle.load(f)
