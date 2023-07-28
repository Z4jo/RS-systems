import os
import sys
sys.path.append('../contet-based/')
sys.path.append('../hybrid/')
sys.path.append('../collaborative-filtering/')
import linear_regression
import mnl_regression_demo
import linear_regression_demo
import svd
import svd_demo
import pandas as pd
import numpy as np
import pickle

PATH_TO_RATINGS='../../data_movilens/ml-latest-small/ratings.csv'
PATH_TO_MOVIES = '../../data_movilens/ml-latest-small/movies.csv'
PATH_TO_MODEL = '../contet-based/linear_model.pickle'


def main(user_id: int):
    ratings_df = pd.read_csv(PATH_TO_RATINGS,delimiter = ',')
    movies_df = pd.read_csv(PATH_TO_MOVIES,delimiter = ',')
    rating_matrix= pd.pivot_table(data=ratings_df,index="userId",columns="movieId", values="rating")

    PATH_TO_WEIGHTED_MODEL = '../hybrid/weighted_model.pickle'
    PATH_TO_LINEAR_MODEL = '../contet-based/linear_model.pickle'
    PATH_TO_MNL_MODEL = '../contet-based/mnl_model.pickle'

    ratings_df = pd.read_csv(PATH_TO_RATINGS,delimiter = ',')
    movies_df = pd.read_csv(PATH_TO_MOVIES,delimiter = ',')
    rating_matrix= pd.pivot_table(data=ratings_df,index="userId",columns="movieId", values="rating")

    weighted_model = 0
    linear_model = 0
    mnl_model = 0

    if os.path.exists(PATH_TO_WEIGHTED_MODEL) and os.path.exists(PATH_TO_LINEAR_MODEL) and os.path.exists(PATH_TO_MNL_MODEL):
        with open(PATH_TO_WEIGHTED_MODEL,"rb") as file:
            weighted_model = pickle.load(file)
        with open(PATH_TO_LINEAR_MODEL,"rb") as file:
            linear_model = pickle.load(file)
        with open(PATH_TO_MNL_MODEL,"rb") as file:
            mnl_model = pickle.load(file)
    else:
        raise OSError("path for the model wasn't found")

    user_count,item_count = rating_matrix.shape
    ud = linear_regression.generate_user_dataframe(rating_matrix,movies_df,user_id)

    numbers_arr = [i for i in range(0,item_count)]
    rating_matrix.columns = numbers_arr
    rating_matrix_clone = rating_matrix.copy()
    rating_matrix_clone = rating_matrix.reset_index(drop = True)
    rating_matrix_clone.index.name = "userId"

    linear_y = linear_regression_demo.prediction(ud,linear_model,user_id)
    mnl_y = mnl_regression_demo.prediction(ud,mnl_model,user_id)
    X = [linear_y,mnl_y]
    y_pred = 0
    for i,x in enumerate(X):
        y_pred += np.dot(x,weighted_model[i])

    nan_indexes = rating_matrix.iloc[user_id].index[rating_matrix.iloc[user_id].isna()] 
    if y_pred.shape == nan_indexes.shape:
        rating_matrix_clone.iloc[user_id,nan_indexes] = y_pred
    svd_y = svd.predict(rating_matrix_clone) 
    svd_y_cleared = svd_demo.clear_result(user_id,svd_y,rating_matrix)
    
    return svd_y_cleared.iloc[user_id]
    

if __name__ == '__main__':

    main(0)
