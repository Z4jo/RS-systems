import math
import numpy as np
import random as rnd
import pickle
import pandas as pd

PATH_TO_DATA = '../../data_movilens/ml-latest-small/ratings.csv'

def get_indexes_of_not_empty_ratings_by_user(rating_matrix):
    not_emnpty_column_vectors = []
    for row in range(len(rating_matrix)): 
        not_empty_indexes=[] 
        for column,_ in enumerate(rating_matrix.columns):
            cell_value=rating_matrix.iloc[row,column]
            if not np.isnan(cell_value):
                not_empty_indexes.append((row,column)) 
        not_emnpty_column_vectors.append(not_empty_indexes)
    return not_emnpty_column_vectors 

def size_of_not_empty_ratings(not_emnty_column_indexes):
    ret = 0 
    for user_indexes in not_emnty_column_indexes:
        ret+=len(user_indexes)
    return ret

def create_parts_dataset(k_size, random_seed, rating_matrix):
    parts = []
    rnd.seed(random_seed)
    #TODO: upravit pickle load
    #not_empty_ratings_indexes=get_indexes_of_not_empty_ratings_by_user(rating_matrix)
    not_empty_ratings_indexes=0
    with open('../cross_validation/column_vectors.pickle', 'rb') as file:
        not_empty_ratings_indexes=pickle.load(file)
        #pickle.dump(not_empty_ratings_indexes,file)
    percentage_value = 1 / k_size
    for _ in range(k_size):
        part = []
        for row in not_empty_ratings_indexes:
            size_to_extract =  math.floor(len(row) * percentage_value)
            for _ in range(size_to_extract):
                row_clone = row[:]
                random_number = rnd.randint(0, len(row)-1)
                row_index,column_index = row_clone.pop(random_number)
                part.append((row_index,column_index,rating_matrix.iloc[row_index,column_index]))
        parts.append(part)
    return parts

if __name__ == '__main__':
    dataframe=pd.read_csv(PATH_TO_DATA,delimiter=',')
    rating_matrix= pd.pivot_table(data=dataframe,index="userId",columns="movieId", values="rating")
    parts = create_parts_dataset(5,131,rating_matrix)      
    for part in parts:
        print(len(part))
    
