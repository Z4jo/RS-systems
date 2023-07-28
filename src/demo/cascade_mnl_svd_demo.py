import os
import sys
sys.path.append('../contet-based/')
sys.path.append('../collaborative-filtering/')
sys.path.append('../hybrid/')
import mnl_regression
import pandas as pd
import numpy as np
import pickle
import svd
import svd_demo
PATH_TO_RATINGS='../../data_movilens/ml-latest-small/ratings.csv'
PATH_TO_MOVIES = '../../data_movilens/ml-latest-small/movies.csv'
PATH_TO_WEIGHTED_MODEL = '../hybrid/weighted_model.pickle'
PATH_TO_MNL_MODEL = '../contet-based/mnl_model.pickle'

def unclean_predict(rating_matrix,ud,model,user_id):
    rating_matrix_clone = rating_matrix.copy()
    numbers_array = [num for num in range(0, rating_matrix_clone.shape[1])]
    rating_matrix_clone.columns = numbers_array
    ud = ud.rename(columns = {"(no genres listed)": "beta0" })
    nan_df = ud[ud['rating'].isna()].drop(["movieId","rating","beta0"],axis=1)
    weights, bias = model[user_id]
    y_pred = mnl_regression.calculate_prediction(nan_df, weights, bias)
    nan_indexes = rating_matrix_clone.iloc[user_id].index[rating_matrix_clone.iloc[user_id].isna()] 
    if y_pred.shape == nan_indexes.shape:
        rating_matrix_clone.iloc[user_id,nan_indexes] = y_pred
    return rating_matrix_clone

def main(user_id: int):
    ratings_df = pd.read_csv(PATH_TO_RATINGS,delimiter = ',')
    movies_df = pd.read_csv(PATH_TO_MOVIES,delimiter = ',')
    rating_matrix= pd.pivot_table(data=ratings_df,index="userId",columns="movieId", values="rating")

    mnl_model = 0

    if os.path.exists(PATH_TO_MNL_MODEL):
        with open(PATH_TO_MNL_MODEL,"rb") as file:
            mnl_model = pickle.load(file)
    else:
        raise OSError("path for the model wasn't found")

    user_count,item_count = rating_matrix.shape
    ud = mnl_regression.generate_user_dataframe(rating_matrix,movies_df,user_id)

    numbers_arr = [i for i in range(0,item_count)]
    rating_matrix.columns = numbers_arr
    rating_matrix_clone = rating_matrix.reset_index(drop = True)
    rating_matrix_clone.index.name = "userId"
    
    mnl_y = unclean_predict(rating_matrix,ud,mnl_model,user_id)
    svd_y = svd.predict(mnl_y)    
    svd_y_clean = svd_demo.clear_result(user_id,svd_y,rating_matrix)

    return svd_y_clean.iloc[user_id]
    


if __name__ == '__main__':
    main(0)

 
