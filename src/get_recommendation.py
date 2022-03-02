import argparse
import pickle
import json
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="input json path",
                        type=str)
    parser.add_argument("-m", "--recommendation-matrix-path", type=str, default="data/recommender_results/matrix.pkl")
    args = parser.parse_args()
    return args.data, args.recomendation_matrix_path


def serve_recommendations(userId, recommend_matrix, users, prods, new_prods, best_prods):
    if userId in users.index:
        bought = recommend_matrix.loc[recommend_matrix["userId"] == userId].productId.values.tolist()
        best = recommend_matrix.loc[users[userId], :].sort_values(ascending=False)[:(100+len(bought))].index.values.tolist()
        best100 = [prods[prod] for prod in best if prod not in bought]
    else:
        best100 = best_prods
    arang = [i for i in range(10)]
    recommendation = []
    for i in range(10):
        j = random.choice(arang)
        recommendation.append(best100[i * 10 + j])
    recommendation.insert(random.choice(arang), random.choice(new_prods))
    return recommendation


if __name__ == "__main__":
    # TODO
    data, recommendation_matrix_path = parse_args()
    request = json.loads(data)
    with open(recommendation_matrix_path, "rb") as f:
        recommend_matrix = pickle.load(f)
    users = None
    prods = None
    new_products = None
    best_products = None
    userId = None
    output_path = None
    recommendations = serve_recommendations(userId, recommend_matrix, users, prods, new_products, best_products)
    with open(output_path, "wb") as f:
        pickle.dump(recommendations, f)
