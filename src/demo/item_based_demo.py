import sys
sys.path.append('../collaborative-filtering/')
import neighbour_methods
import pandas as pd
import numpy as np
sys.path.append('../math_functions/')
import sim_func


PATH_TO_DATA='../../data_movilens/ml-latest-small/ratings.csv'
def mean_center_matrix(rating_matrix): 
    mean = rating_matrix.mean(1)
    mean_centered_rm = rating_matrix.sub(mean, axis=0)
    return mean_centered_rm 

def main(user_id :int):
    dataframe = pd.read_csv(PATH_TO_DATA,delimiter=',')
    rating_matrix = pd.pivot_table(data=dataframe,index="userId",columns="movieId", values="rating")
    rating_matrix = rating_matrix.reset_index(drop = True)
    user_count,item_count = rating_matrix.shape 
    rating_matrix.index.name = "userId"
    numbers_array = [num for num in range(0, item_count)]
    rating_matrix.columns = numbers_array

    mean_centered_matrix = mean_center_matrix(rating_matrix.copy())

    similarity_matrix = [[0 for _ in range(item_count)] for _ in range(item_count)]   
    intersections_matrix = [[0 for _ in range(item_count)] for _ in range(item_count)]   
    user_series = mean_centered_matrix.iloc[user_id]

    active_user_nan_index = user_series.index[user_series.isna()].values
    result_series = pd.Series([np.nan] * item_count)

    rng = np.random.default_rng(seed=42)
    test = rng.random((3, 3))
    print(test)
    cov_matrix = np.corrcoef(test)

    print(cov_matrix)
"""
    for nan_index in active_user_nan_index:
        print(nan_index)
        active_column = mean_centered_matrix[nan_index]
        active_column_not_nan_indexes = active_column.index[active_column.notna()].values
        for item_j in range(item_count):
            if type(intersections_matrix[item_j][nan_index]) == float: 
                continue

            if  similarity_matrix[item_j][nan_index] != 0:
                continue

            column_j = mean_centered_matrix[item_j]
            column_j_not_nan_indexes = column_j.index[column_j.notna()].values
            intersection = np.intersect1d(column_j_not_nan_indexes,active_column_not_nan_indexes)
            if len(intersection) > 4 and pd.notna(column_j.iloc[user_id]):

                column_j_ratings = column_j.iloc[intersection]
                active_column_ratings = active_column.iloc[intersection]
                similarity = sim_func.adjusted_cosine(column_j_ratings.values,active_column_ratings.values)
                similarity_matrix[nan_index][item_j] = similarity
                similarity_matrix[item_j][nan_index] = similarity 
                intersections_matrix[nan_index][item_j] = intersection
                intersections_matrix[item_j][nan_index] = intersection
            else:
                similarity_matrix[nan_index][item_j] = np.nan
                similarity_matrix[item_j][nan_index] = np.nan
                intersections_matrix[nan_index][item_j] = np.nan
                intersections_matrix[item_j][nan_index] = np.nan

    for nan_index in active_user_nan_index:
        numerator = 0
        denominator = 0
        for item_j,intersection in enumerate(intersections_matrix[nan_index]):

            item_j_not_nan_indexes = rating_matrix[item_j].index[rating_matrix[item_j].notna()].values

            if type(intersection) != np.ndarray or nan_index not in item_j_not_nan_indexes :
                continue
            numerator += similarity_matrix[nan_index][item_j] * rating_matrix.iloc[user_id,item_j]
            denominator += abs(similarity_matrix[nan_index][item_j])
        if denominator == 0 or numerator == 0:
            continue
        result_series.iloc[nan_index] = numerator/denominator
        
    print(result_series)



"""
if __name__ == '__main__':
    ret = main(0)    
