import argparse
import pickle

def mean_imputation(df):  # TODO
    return df

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", help="na imputation method, one of: ...",
                        type=int)
    parser.add_argument("-i", "--input-path-df", help="input pickle file path df",
                        type=str)
    parser.add_argument("-j", "--input-path-pkl", help="input pickle file path pkl",
                        type=str)
    parser.add_argument("-o", "--output-path", help="output pickle file path pkl",
                        type=str)
    args = parser.parse_args()
    return args.method, args.input_path_df, args.input_path_pkl, args.output_path


if __name__ == "__main__":
    method, input_path_df, input_path_pkl, output_path = parse_args()
    with open(input_path_df, "rb") as f:
        df_clean = pickle.load(f)
    with open(input_path_pkl, "rb") as f:
        results = pickle.load(f)

    df_clean["score"] = results

    method_functions = {"mean": mean_imputation}
    # make recommendation matrix
    assert method in method_functions.keys(), f"Unrecognised method: {method}"
    recommendation_matrix = method_functions[method](df_clean)

    with open(output_path, "wb") as f:
        pickle.dump(recommendation_matrix, f)
