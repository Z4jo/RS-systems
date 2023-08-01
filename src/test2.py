import numpy as np
import pandas as pd
import pickle
import sys
import os
import multiprocessing 
import cProfile
import pstats
sys.path.append('./cross_validation/')
sys.path.append('./data_procession/')
sys.path.append('./math_functions/')
import sim_func
import cross_validation

#PATH_TO_DATA='../data_movilens/test.csv'
PATH_TO_DATA='../data_movilens/ml-latest-small/ratings.csv'

if __name__ == '__main__':
    dataframe = pd.read_csv(PATH_TO_DATA,delimiter=',')
    rating_matrix = pd.pivot_table(data=dataframe,index="userId",columns="movieId", values="rating")
    rating_matrix = rating_matrix.reset_index(drop = True)
    rating_matrix.index.name = "userId"
    numbers_array = [num for num in range(0, rating_matrix.shape[1])]
    rating_matrix.columns = numbers_array
    users_count = rating_matrix.shape[0]
    film_count = rating_matrix.shape[1]
    parts = []
   
    if not os.path.exists("./cross_validation_parts.pickle"):
        parts = cross_validation.create_parts_dataset(5,131,rating_matrix)
        with open("cross_validation_parts.pickle","wb") as file:
            pickle.dump(parts,file)
    else:
        with open("cross_validation_parts.pickle","rb") as file:
            parts = pickle.load(file)
    
    for iteration, part in enumerate(parts):
        updated_matrix = rating_matrix.copy()
        for rating_tuple in part:
            row,column,rating=rating_tuple
            updated_matrix.iloc[row,column] = np.nan

        users_count = rating_matrix.shape[0]
        film_count = rating_matrix.shape[1]
        
        result_matrix = pd.DataFrame(np.nan, index = range(users_count), columns = range(film_count))
        similarity_matrix = np.full((users_count,users_count),0.0)
        not_nan_indexes_mask = rating_matrix.notna()
        nan_indexes_mask = rating_matrix.isna()
        for user_i in range(users_count):
            rating_matrix_clone = updated_matrix.copy()
            active_user_not_nan_indexes = rating_matrix_clone.iloc[user_i].index[not_nan_indexes_mask.iloc[user_i]].values
            active_user_nan_indexes = rating_matrix_clone.iloc[user_i].index[nan_indexes_mask.iloc[user_i]].values
            active_user_mean = rating_matrix_clone.iloc[user_i].mean()
            for user_j in range(users_count):
                if similarity_matrix[user_j,user_i]!=0:
                    continue
                user_j_not_nan_indexes = rating_matrix_clone.iloc[user_j].index[not_nan_indexes_mask.iloc[user_j]].values
                intersection = np.intersect1d(user_j_not_nan_indexes,active_user_not_nan_indexes)
                ratings_j = rating_matrix_clone.iloc[user_j,intersection].values
                ratings_i = rating_matrix_clone.iloc[user_i,intersection].values 
                if len(ratings_i) > 4:
                    similarity = sim_func.pearson_coefficient(ratings_j,ratings_i)
                    similarity_matrix[user_i,user_j] = similarity
                    similarity_matrix[user_j,user_i] = similarity
            
            active_user_nan_indexes = rating_matrix_clone.iloc[user_i].index[nan_indexes_mask.iloc[user_i]].values
            for index in active_user_nan_indexes:
                numerator = 0
                denominator = 0
                for user_j in range(users_count):
                    if similarity_matrix[user_i,user_j] == 0:
                        continue
                    if not_nan_indexes_mask.iloc[user_j,index] == True:
                        mean = rating_matrix_clone.iloc[user_j].mean()
                        numerator += (rating_matrix_clone.iloc[user_j,index] - mean) * similarity_matrix[user_i,user_j]
                        denominator += abs(similarity_matrix[user_i,user_j])
                if denominator == 0:
                    continue
                result_matrix.iloc[user_i,index] = active_user_mean+(numerator/denominator)
                    

        with open("./user_based_result"+str(iteration)+".pickle","wb") as file:
            pickle.dump(result_matrix,file)
        print(f"iteration done:{iteration}")
            
