import os
import sys
sys.path.append('../contet-based/')
import linear_regression
import pandas as pd
import numpy as np
import pickle

PATH_TO_RATINGS='../../data_movilens/ml-latest-small/ratings.csv'
PATH_TO_MOVIES = '../../data_movilens/ml-latest-small/movies.csv'
PATH_TO_MODEL = '../contet-based/linear_model.pickle'

def prediction(ud,model,user_id):
    ud = ud.rename(columns = {"(no genres listed)": "beta0" })
    nan_df= ud[ud['rating'].isna()].drop(["movieId","rating","beta0"],axis=1)
    coefficients, bias = model[user_id]
    y_pred = np.dot(nan_df, coefficients) + bias
    return y_pred

def main(user_id: int):
    ratings_df = pd.read_csv(PATH_TO_RATINGS,delimiter = ',')
    movies_df = pd.read_csv(PATH_TO_MOVIES,delimiter = ',')
    rating_matrix= pd.pivot_table(data=ratings_df,index="userId",columns="movieId", values="rating")

    model = 0

    if os.path.exists(PATH_TO_MODEL):
        with open(PATH_TO_MODEL,"rb") as file:
            model = pickle.load(file)
    else:
        raise OSError("path for the model wasn't found")

    user_count,item_count = rating_matrix.shape
    ud = linear_regression.generate_user_dataframe(rating_matrix,movies_df,user_id)

    numbers_arr = [i for i in range(0,item_count)]
    rating_matrix.columns = numbers_arr
    rating_matrix_clone = rating_matrix.reset_index(drop = True)
    rating_matrix_clone.index.name = "userId"

    result_series = pd.Series([np.nan] * item_count)

    y_pred = prediction(ud,model,user_id)
    nan_indexes = rating_matrix.iloc[user_id].index[rating_matrix.iloc[user_id].isna()] 
    if y_pred.shape == nan_indexes.shape:
        result_series.iloc[nan_indexes] = y_pred
    
    return(result_series)
    

if __name__ == '__main__':

    main(0)
