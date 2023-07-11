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
            print(i)
            return weights, bias
        last_mse = test
        
        dw = (1/len(y)) * np.dot(X.T,(y_pred - y))
        db = (1/len(y)) * np.sum(y_pred - y)
        weights = weights - l_rate * dw
        bias = bias - l_rate * db
    return weights, bias

    

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
    genre_ud = ud.drop(["movieId","rating","beta0"],axis=1)
    #genre_ud['beta0'] = 1
    print(genre_ud)
   # print(genre_ud)
    coefficients, bias = fit(genre_ud, ud_ratings)
    print(coefficients)
    y_pred = np.dot(genre_ud, coefficients) + bias
    
    msee = mse(y_pred,ud_ratings)
    print(msee)
    """
    final_model = gradiant_descent_l(ud_ratings,genre_ud)
    final_prediction = []
    print(final_model)
    for row in genre_ud.iterrows():
        prediction = 0
        for i,parameter in enumerate(row[1]):
            prediction += final_model[i] * parameter
        final_prediction.append(prediction)
    final_prediction = np.array(final_prediction)
    final_prediction = final_prediction.astype(np.float64)
    total = 0
    for i, rating in enumerate(ud_ratings):
        total += (round(final_prediction[i],4) - rating)**2
    rmse = total * (1/len(ud_ratings))
    print(math.sqrt(rmse))

        
        
    for i,rating in enumerate(ud_ratings):
        print(f"actual_rating:{rating},predicted_rating:{round(final_prediction[i],4)}")
    




    #print(final_model)

    #y = np.array([0, 2, 2, 0, 1])
    #x = np.array([[0.13, 0.1, 1.3, -1.27, -2.33], [-0.13, -0.54, 0.95, -0.62, -0.22], [0.64, 0.36, -0.7, 0.04, -1.25]])
    #x = np.array([[0.13, -0.13, 0.64], [0.1, -0.54, 0.36], [1.3, 0.95, -0.7], [-1.27, -0.62, 0.04], [-2.33, -0.22, -1.25]])
    #result = loss_function(y,x)

    user_data = generate_user_dataframe(rating_matrix_clone,movies_df,20)
    ud = user_data.copy()
    ud = ud.dropna()
    ud = ud.rename(columns = {"(no genres listed)": "beta0" })
    ud_ratings = ud['rating']
    genre_ud = ud.drop(["movieId","rating"],axis=1)
        print(result)
    
       

         
"""
