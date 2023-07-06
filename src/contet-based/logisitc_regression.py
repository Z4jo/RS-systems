import numpy as np 
import pandas as pd
import math 
import pickle 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
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

def gradiant_descent(step_change, number_of_iterations, parameters, ratings, predictions):
    iter = 0 
    while iter < number_of_iterations:
        updated_parameters = []
        loss_value = loss_function(ratings, predictions)
        for parameter in parameters:
            updated_parameter = parameter - step_change * loss_value
            updated_parameters.append(updated_parameter)
            
        iter += 1

def softmax(predictions):
    exp_vals = np.exp(predictions)
    result = []
    for i in exp_vals:
        sum = np.sum(i)
        result.append(i/sum)
    return result



def loss_function(ratings, predictions):
    softmax_values = softmax(predictions)
    softmax_values = np.log(softmax_values)
    print(softmax_values)
    chosen_values = [] 
    for i,rating in enumerate(ratings):
        #WARNING: ADD -1 TO RATING
        chosen_values.append(softmax_values[i][rating])
    print(chosen_values)
    return np.sum(chosen_values)
        

if __name__ == '__main__':
    ratings_df = pd.read_csv(PATH_TO_RATINGS,delimiter = ',')
    movies_df = pd.read_csv(PATH_TO_MOVIES,delimiter = ',')
    rating_matrix= pd.pivot_table(data=ratings_df,index="userId",columns="movieId", values="rating")
    rating_matrix = rating_matrix.reset_index(drop = True)
    rating_matrix.index.name = "userId"
    
    rating_matrix_clone = rating_matrix.copy()
    y = np.array([0, 2, 2, 0, 1])
    #x = np.array([[0.13, 0.1, 1.3, -1.27, -2.33], [-0.13, -0.54, 0.95, -0.62, -0.22], [0.64, 0.36, -0.7, 0.04, -1.25]])
    x = np.array([[0.13, -0.13, 0.64], [0.1, -0.54, 0.36], [1.3, 0.95, -0.7], [-1.27, -0.62, 0.04], [-2.33, -0.22, -1.25]])
    result = loss_function(y,x)
    print(result)

""" 
    user_data = generate_user_dataframe(rating_matrix_clone,movies_df,20)
    ud = user_data.copy()
    ud = ud.dropna()
    ud = ud.rename(columns = {"(no genres listed)": "beta0" })
    ud_ratings = ud['rating']
    genre_ud = ud.drop(["movieId","rating"],axis=1)
    coefficients = np.full(genre_ud.shape[1],0.04)
    genre_ud['beta0'] = 1
    genre_ud = genre_ud.apply(lambda x: x+0.01)
    result = genre_ud * coefficients
    sum = result.sum(axis = 1)
    result = np.power(np.e,sum)
    result = result.apply(lambda x: x/(1+x))
    print(result)
    
       

         
"""
