import argparse
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, NMF, TruncatedSVD, LatentDirichletAllocation, FactorAnalysis, KernelPCA
from sklearn.manifold import TSNE

def PCA_dimred(df_nlp, df_train, df_test):
    scaler = StandardScaler()
    pca = PCA(0.9, random_state=2022)
    data_nlp = pca.fit_transform(scaler.fit_transform(df_nlp[:, 1:]))
    data_train = pca.transform(scaler.transform(df_train[:, 1:]))
    data_test = pca.transform(scaler.transform(df_test[:, 1:]))
    print(f"Shape of training data reduced from {df_nlp[:, 1:].shape} to {data_nlp.shape}.")
    return np.concatenate((df_nlp[:, :1], data_nlp), 1), \
           np.concatenate((df_train[:, :1], data_train), 1), \
           np.concatenate((df_test[:, :1], data_test), 1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", help="na imputation method, one of: ...",
                        type=str)
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
    args = parser.parse_args()
    return args.method, args.input_path_train, args.input_path_test, args.input_path_nlp, \
           args.output_path_train, args.output_path_test, args.output_path_nlp

if __name__ == "__main__":
    method, input_path_train, input_path_test, input_path_nlp, output_path_train, output_path_test, output_path_nlp = parse_args()

    with open(input_path_train, "rb") as f:
        data_train = pickle.load(f)

    with open(input_path_test, "rb") as f:
        data_test = pickle.load(f)

    with open(input_path_nlp, "rb") as f:
        data_nlp = pickle.load(f)

    data_nlp[np.isnan(data_nlp)] = 0
    data_train[np.isnan(data_train)] = 0
    data_test[np.isnan(data_test)] = 0

    method_functions = {"pca": PCA_dimred}

    # vectorise
    assert method in method_functions.keys(), f"Unrecognised method: {method}"
    data_nlp, data_train, data_test = method_functions[method](data_nlp, data_train, data_test)


    with open(output_path_train, "wb") as f:
        pickle.dump(data_train, f)

    with open(output_path_test, "wb") as f:
        pickle.dump(data_test, f)

    with open(output_path_nlp, "wb") as f:
        pickle.dump(data_nlp, f)