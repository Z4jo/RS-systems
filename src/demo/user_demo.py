import sys
sys.path.append('../collaborative-filtering/')
import sim_func
import neighbour_methods
import pandas as pd 
import numpy as np

#PATH_TO_DATA='../../data_movilens/test.csv'
PATH_TO_DATA='../../data_movilens/ml-latest-small/ratings.csv'
def main(user_id: int):
    dataframe=pd.read_csv(PATH_TO_DATA,delimiter=',')
    rating_matrix= pd.pivot_table(data=dataframe,index="userId",columns="movieId", values="rating")
    rating_matrix = rating_matrix.reset_index(drop = True)
    rating_matrix.index.name = "userId"
    numbers_array = [num for num in range(0, rating_matrix.shape[1])]
    rating_matrix.columns = numbers_array
    row = rating_matrix.iloc[user_id]
    missing_indices = [np.where(row.isnull())[0]]
    column_vectors_user =  neighbour_methods.get_indexes_of_not_empty_ratings_by_user(rating_matrix)
    ret = neighbour_methods.predict_ratings_user_based(user_id, missing_indices[0], column_vectors_user, rating_matrix)
    return ret

if __name__ == '__main__':
    ret = main(0)     

