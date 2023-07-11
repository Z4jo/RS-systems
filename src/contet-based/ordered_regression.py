import math 
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

def softmax(predictions):
    exp_vals = np.exp(predictions)
    result = []
    for i in exp_vals:
        total = np.sum(i)
        result.append(i/total)
    return result

def mse(pred_y, y):
    total = 0
    for i in range(len(pred_y)):
        total += (pred_y[i]-y.iloc[i])**2
    return (1/len(pred_y))*total

def logistic_function(coefficients, independent_values, intercept):
    beta_x = np.dot(coefficients, independent_values)
    constant = np.power(np.e, -(intercept - beta_x))
    return 1 / (1+constant)

def logistic_function_derrivative(sigma):
    return sigma * (1 - sigma)

def indicator_function(y, k):
    if y == k:
        return 1
    else:
        return 0

def calculate_predictions(genre_df, coefficients, intercepts):

    probabilities = []
    for _,parameters in genre_df.iterrows():
        x_probability = np.zeros(len(intercepts))
        for j,class_intercept in enumerate(intercepts):
            sigma = logistic_function(coefficients,parameters,class_intercept)
            x_probability[j] = sigma
        probabilities.append(x_probability)
    return probabilities

def coefficients_partial_derrivative(ratings ):
     
    return

def intercept_partial_derrivative():
    return

def fit(X, y):
    return

if __name__ == '__main__':
    ratings_df = pd.read_csv(PATH_TO_RATINGS,delimiter = ',')
    movies_df = pd.read_csv(PATH_TO_MOVIES,delimiter = ',')
    rating_matrix= pd.pivot_table(data=ratings_df,index="userId",columns="movieId", values="rating")
    rating_matrix = rating_matrix.reset_index(drop = True)
    rating_matrix.index.name = "userId"
    
    rating_matrix_clone = rating_matrix.copy()
    user_data = generate_user_dataframe(rating_matrix_clone,movies_df,0)
    ud = user_data.copy()
    ud = ud.dropna()
    ud = ud.rename(columns = {"(no genres listed)": "beta0" })
    ud_ratings = ud['rating']
    genre_df = ud.drop(["movieId","rating","beta0"],axis=1)
    #genre_ud['beta0'] = 1
    print(genre_df)
   # print(genre_ud)
    predictions = calculate_predictions(genre_df, np.ones(19),np.ones(5))
    print(predictions)
    
    """
    coefficients, bias = fit(genre_ud, ud_ratings)
    print(coefficients)
    y_pred = np.dot(genre_ud, coefficients) + bias
    
    msee = mse(y_pred,ud_ratings)
    print(msee)
    """
