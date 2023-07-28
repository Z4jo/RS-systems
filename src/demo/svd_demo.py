import os
import sys
sys.path.append('../collaborative-filtering/')
import svd
import pandas as pd
import numpy as np
import pickle

PATH_TO_RATINGS='../../data_movilens/ml-latest-small/ratings.csv'


def clear_result (user_id, result_df, rating_matrix):
    column_arr = [i for i in range(rating_matrix.shape[1])]
    rating_matrix.columns = column_arr
    mask = rating_matrix.iloc[user_id].index[rating_matrix.iloc[user_id].notna()].values  
    result_df.iloc[user_id,mask] = np.nan
    return result_df

def main(user_id: int):
    ratings_df = pd.read_csv(PATH_TO_RATINGS,delimiter = ',')
    rating_matrix= pd.pivot_table(data=ratings_df,index="userId",columns="movieId", values="rating")

    svd_y = svd.predict(rating_matrix) 
    svd_y_cleared = clear_result(user_id,svd_y,rating_matrix)
    
    return svd_y_cleared.iloc[user_id]
    

if __name__ == '__main__':

    main(0)
