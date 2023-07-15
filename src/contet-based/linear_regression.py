import numpy as np 
import pandas as pd
import pickle 
import os
import sys
import multiprocessing

from pandas.core.api import notnull 
sys.path.append('../cross_validation/')
import cross_validation

#PATH_TO_MOVIES = '../../data_movilens/content-based movies.csv'
#PATH_TO_RATINGS = '../../data_movilens/content-based ratings.csv'
PATH_TO_RATINGS='../../data_movilens/ml-latest-small/ratings.csv'
PATH_TO_MOVIES = '../../data_movilens/ml-latest-small/movies.csv'


def generate_user_dataframe(rating_matrix, movies_df, user_id):
    user_ratings_df = rating_matrix.iloc[user_id]
    user_ratings_df = user_ratings_df.to_frame('rating').reset_index()
    movie_id_indeces = user_ratings_df['movieId']
    filtered_movies_df = movies_df[movies_df['movieId'].isin(movie_id_indeces)]
    genres_df = filtered_movies_df['genres'].str.get_dummies('|')
    genres_id = pd.concat([filtered_movies_df['movieId'],genres_df], axis = 1)
    merged_df = pd.merge(genres_id, user_ratings_df, on='movieId', how='left')
    return merged_df

def mse(pred_y, y):
    total = 0
    for i in range(len(pred_y)):
        total += (pred_y[i]-y.iloc[i])**2
    return (1/len(pred_y))*total
        
def fit(X, y):
    weights = np.zeros(X.shape[1])
    bias = 0
    l_rate = 0.01
    last_mse = 1000
    for i in range(6000):
        y_pred = np.dot(X, weights) + bias
        test = mse(y_pred, y)
        if last_mse - test < 0.00001: 
            return weights, bias
        last_mse = test
        
        dw = (1/len(y)) * np.dot(X.T,(y_pred - y))
        db = (1/len(y)) * np.sum(y_pred - y)
        weights = weights - l_rate * dw
        bias = bias - l_rate * db
    return weights, bias

def get_indexes_of_not_empty_ratings_by_user(df):
    array_of_arrays = []
    for _,row in df.iterrows():
        non_nan_columns = [column for column, value in row.items() if pd.notnull(value)]
        array_of_arrays.append(non_nan_columns)
    return array_of_arrays
   

def user_profile_prediction(ud,index):
    ud = ud.rename(columns = {"(no genres listed)": "beta0" })
    movies_id= ud[ud['rating'].isna()]
    movies_id= movies_id['movieId']
    nan_df= ud[ud['rating'].isna()].drop(["movieId","rating","beta0"],axis=1)
    ud = ud.dropna()
    ud_ratings = ud['rating']
    genre_ud = ud.drop(["movieId","rating","beta0"],axis=1)

    coefficients, bias = fit(genre_ud, ud_ratings)
    y_pred = np.dot(nan_df, coefficients) + bias
    return (y_pred,index)

def predict(rating_matrix,movies_df):
    rating_matrix = rating_matrix.reset_index(drop = True)
    rating_matrix.index.name = "userId"

    iterable = []
    for i,_ in rating_matrix.iterrows():
        user_dataframe = generate_user_dataframe(rating_matrix,movies_df,i)
        iterable.append((user_dataframe,i))

    pool = multiprocessing.Pool(processes=8)
    users_predictions = pool.starmap(user_profile_prediction, iterable)
    pool.close()
    pool.join()
    #rating_matrix.rename_axis(range(len(rating_matrix.columns)), axis=1, inplace=True)
    for predictions in users_predictions: 
        #NOTE: predictions[0] = y predictions ; predictions[1] = user_index
        numbers_array = [num for num in range(0, rating_matrix.shape[1])]
        rating_matrix.columns = numbers_array
        nan_indexes = rating_matrix.iloc[predictions[1]].index[rating_matrix.iloc[predictions[1]].isna()] 
        if nan_indexes.shape != np.array(predictions[0]).shape:
            raise ValueError('nan_indexes and predictions arrays are not matching')
        for i,prediction in enumerate(predictions[0]):
            rating_matrix.iloc[predictions[1],nan_indexes[i]] = prediction
    
    return rating_matrix
    
        
def get_nan_indexes(df):
    non_nan_indexes = []

    for row_idx, row in enumerate(df.index):
        for col_idx, col in enumerate(df.columns):
            if not pd.isna(df.loc[row, col]):
                non_nan_indexes.append((row_idx, col_idx))

    return non_nan_indexes

if __name__ == '__main__':
    ratings_df = pd.read_csv(PATH_TO_RATINGS,delimiter = ',')
    movies_df = pd.read_csv(PATH_TO_MOVIES,delimiter = ',')
    rating_matrix= pd.pivot_table(data=ratings_df,index="userId",columns="movieId", values="rating")
    parts = []
    if not os.path.exists("./cross_validation_parts.pickle"):
        parts = cross_validation.create_parts_dataset(5,131,rating_matrix)
        with open("cross_validation_parts.pickle","wb") as file:
            pickle.dump(parts,file)
    else:
        with open("cross_validation_parts.pickle","rb") as file:
            parts = pickle.load(file)

    for iteration,part in enumerate(parts):
        rating_matrix_clone = rating_matrix.copy()
        for rating_tuple in part:
            row,column,rating=rating_tuple
            rating_matrix_clone.iloc[row,column] = rating
        result = predict(rating_matrix_clone,movies_df)
        numbers_array = [num for num in range(0, rating_matrix.shape[1])]
        rating_matrix_clone.columns = numbers_array
        not_nan_indexes = get_nan_indexes(rating_matrix_clone)
        for nan_indexes in not_nan_indexes:
            result.iloc[nan_indexes[0], nan_indexes[0]]= np.nan

        with open('../linear_regression'+str(iteration)+'.pickle', 'wb') as file:
            pickle.dump(result,file)
        print(f"iteration done:{iteration}")
            
    
