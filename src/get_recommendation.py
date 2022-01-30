import argparse
import pickle
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="input json path",
                        type=str)
    parser.add_argument("-m", "--recomendation-matrix-path", type=str, default="data/recommender_results/matrix.pkl")
    args = parser.parse_args()
    return args.data, args.recomendation_matrix_path


if __name__ == "__main__":
    data, recomendation_matrix_path = parse_args()
    request = json.loads(data)
    with open(recomendation_matrix_path, "rb") as f:
        recomend_matrix = pickle.load(f)

    # TODO
