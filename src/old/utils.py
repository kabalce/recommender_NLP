import pandas as pd
import numpy as np
import copy


def load_reviews_to_df(path: str, scale: bool=False, remove_duplicates: bool=False) -> pd.DataFrame:
    """
    Loads reviews from file to DataFrame. 
    """
    reviews = pd.read_csv(path)
    reviews['score'] = reviews['score'].astype(float)
    reviews = reviews[['productId', 'userId', 'score']]
    reviews['score'] = reviews[['score']].astype(float)

    if scale:
        reviews[['score']] = scaling(reviews['score'])

    if remove_duplicates:
        reviews = reviews.groupby(['userId', 'productId'], as_index=False)['score'].mean()
    
    return reviews[reviews['userId'] != 'unknown']


def load_score_data(path: str, fill_na: bool=False) -> pd.DataFrame:
    """
    Creates matrix corresponding to reviews. The reviews are stored in a specified file.   
    """
    reviews = load_reviews_to_df(path, scale=False, remove_duplicates=True)
    
    products = pd.DataFrame({
        'productId': reviews['productId'].drop_duplicates(),
        'productIndex': range(len(reviews['productId'].drop_duplicates()))
    })

    users = pd.DataFrame({
        'userId': reviews['userId'].drop_duplicates(),
        'userIndex': range(len(reviews['userId'].drop_duplicates()))
    })

    scores = reviews.merge(users, on="userId")
    scores = scores.merge(products, on="productId")

    scores = scores.pivot(
        index='userId',
        columns='productId',
        values='score'
        ).values

    if fill_na:
        scores[np.isnan(scores)] = 0

    return scores


def train_test_split(ratings, train_size: float=0.8, fill:bool=True):

    validation = np.empty(ratings.shape)
    validation.fill(np.nan)
    train = ratings.copy()
    ratings[np.isnan(ratings)] = 0

    for user in np.arange(ratings.shape[0]):
        no_ratings = len(ratings[user,:].nonzero()[0])
        val_ratings = np.random.choice(
            ratings[user, :].nonzero()[0],
            size=int(train_size*no_ratings),
            replace=False
        )
        train[user, val_ratings] = np.nan
        validation[user, val_ratings] = ratings[user, val_ratings]

    inds = np.where(np.isnan(train))
    
    if fill:
        train[inds] = np.nansum([
                                 np.take(np.nanmean(train, axis=1), inds[0]), 
                                 np.take(np.nanmean(train, axis=0), inds[1])], axis=0)/2
    else:
        train[inds] = 0
        
    return train, validation


def rmse(prediction, ground_truth):
    rss = np.nansum((prediction - ground_truth)**2)
    return(np.sqrt(rss/np.count_nonzero(~np.isnan(ground_truth))))