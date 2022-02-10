import pandas as pd
import numpy as np

def rmse(prediction, ground_truth):
    rss = np.nansum((prediction - ground_truth)**2)
    return(np.sqrt(rss/np.count_nonzero(~np.isnan(ground_truth))))
