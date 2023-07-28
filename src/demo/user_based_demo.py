import pandas as pd 
import numpy as np
import sys
sys.path.append('../math_functions/')
import sim_func
PATH_TO_DATA='../../data_movilens/ml-latest-small/ratings.csv'

def main(user_id :int):
    dataframe=pd.read_csv(PATH_TO_DATA,delimiter=',')
    rating_matrix= pd.pivot_table(data=dataframe,index="userId",columns="movieId", values="rating")
    rating_matrix = rating_matrix.reset_index(drop = True)
    rating_matrix.index.name = "userId"
    numbers_array = [num for num in range(0, rating_matrix.shape[1])]
    rating_matrix.columns = numbers_array


    user_series = rating_matrix.iloc[user_id]
    print(user_series.shape)
    active_user_mean = user_series.mean()

    active_user_nan_index = user_series.index[user_series.isna()].values
    active_user_not_nan_index = user_series.index[user_series.notna()].values

    similarities = pd.Series([[0]]*rating_matrix.shape[0])
    intersections =  pd.Series([[0]]*rating_matrix.shape[0])
    result_series = pd.Series([np.nan] * user_series.shape[0])

    for user_j in range(rating_matrix.shape[0]):
        
        user_j_not_nan_indexes = rating_matrix.iloc[user_j].index[rating_matrix.iloc[user_j].notna()].values
        intersection = np.intersect1d(active_user_not_nan_index,user_j_not_nan_indexes)
        if len(intersection) < 5:
            continue
        intersections.iloc[user_j] = intersection
        active_user_ratings = rating_matrix.iloc[user_id,intersection].values 
        user_j_ratings = rating_matrix.iloc[user_j,intersection].values 
        similarity = sim_func.pearson_coefficient(user_j_ratings,active_user_ratings)
        similarities.iloc[user_j] = similarity
    for nan_index in active_user_nan_index:
        print(nan_index)
        numerator = 0
        denominator = 0
        for user_j, intersection in enumerate(intersections):
            user_j_not_nan_indexes = rating_matrix.iloc[user_j].index[rating_matrix.iloc[user_j].notna()].values
            if len(intersection) < 2 :
                continue
            if nan_index in user_j_not_nan_indexes:
                mean = rating_matrix.iloc[user_j].mean()
                numerator += similarities.iloc[user_j] * (rating_matrix.iloc[user_j,nan_index] - mean)
                denominator += abs(similarities.iloc[user_j])
            
        if denominator == 0 or numerator == 0:
            continue
        result_series.iloc[nan_index] = (numerator/denominator) + active_user_mean 
                        
    return result_series



if __name__ == '__main__':
    ret = main(1)     
    print(ret)

    
