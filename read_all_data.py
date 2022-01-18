import gzip
import pandas as pd
import argparse


# Excluded from processing:
# - "Books", 4.7 GB
# - "Movies_&_TV", 2.9 GB
# - "Music", 2.2 GB
categories = ["Amazon_Instant_Video", "Arts", "Automotive", "Baby", "Beauty", "Cell_Phones_&_Accessories",
              "Clothing_&_Accessories", "Electronics", "Gourmet_Foods", "Health", "Home_&_Kitchen",
              "Industrial_&_Scientific", "Jewelry", "Kindle_Store", "Musical_Instruments", "Office_Products",
              "Patio", "Pet_Supplies", "Shoes", "Software", "Sports_&_Outdoors", "Tools_&_Home_Improvement",
              "Toys_&_Games", "Video_Games", "Watches"]


def load_reviews_to_df(path) -> pd.DataFrame:
    reviews_array = []
    dictionary = {}
    with gzip.open(path) as raw_data:
        for review in raw_data:
            this_line = review.decode("utf-8").split(":")
            if len(this_line) > 1:
                dictionary[this_line[0]] = this_line[1].strip()
            else:
                reviews_array.append(dictionary)
                dictionary = {}

    colNames = ['productId', 'title', 'price', 'userId',
                'profileName', 'helpfulness', 'score',
                'time', 'summary', 'text']

    reviews = pd.DataFrame(reviews_array)
    reviews.columns = colNames
    reviews[['score']] = reviews[['score']].astype(float)
    reviews['time'] = pd.to_datetime(reviews['time'], unit='s')
    return reviews

def save_reduced_reviews(input, output):
    reviews = load_reviews_to_df(input)
    reviews["helpfulness_num"] = reviews["helpfulness"].apply(lambda x: int(x.split("/")[0]))
    reviews["helpfulness_den"] = reviews["helpfulness"].apply(lambda x: int(x.split("/")[1]))
    reviews.to_csv(output, index=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-dir", help="compressed files directory", type=str)
    args = parser.parse_args()
    return args.source_dir

if __name__ == "__main__":
    source = parse_args()
    for category in categories:
        print(category)
        input_path = f'{source}/{category}.txt.gz'
        output_path = f'data/raw/{category}.csv'
        save_reduced_reviews(input_path, output_path)
