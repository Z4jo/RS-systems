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
    return np.array(result)

def mse(pred_y, y):
    total = 0
    for i in range(len(pred_y)):
        total += (pred_y[i]-y.iloc[i])**2
    return (1/len(pred_y))*total

def logistic_function(coefficients, independent_values, intercept):
    beta_x = np.dot(coefficients, independent_values)
    constant = np.power(np.e, (-intercept + beta_x))
    return 1 / (1+constant)

def onehot(ratings, num_classes):
    encoded = pd.get_dummies(ratings)
    encoded = encoded.astype(int)
    # Add missing columns if ratings contain classes not present in the data
    return encoded.values

def fit(X, y,class_number, iter):
    weights = np.zeros((X.shape[1],class_number))
    bias = np.ones((class_number))
    l_rate = 0.001
    last_mse = 1000
    print(weights.shape)
    print(X.shape)
    for i in range(iter):
        y_pred = np.dot(X,weights) + bias
        test = mse(y_pred[0], y)
        #if last_mse - test < 0.00000000001: 
        #    print(math.sqrt(test))
        #    return weights, bias, test 
        last_mse = test
        dw = (1/X.shape[0])*np.dot(X.values.T,softmax(y_pred) - onehot(y, class_number))
        db = (1/X.shape[0])*np.sum(softmax(y_pred) - onehot(y, class_number))
        weights = weights - l_rate * dw
        bias = bias - l_rate * db
        loss = -np.mean(np.log(softmax(y_pred)[np.arange(len(y)), np.vectorize(lambda x: x - 1)(y)]))
        print(f"loss:{loss}; iter:{i}")

    return weights, bias, last_mse

def predict(X, weights, bias):
    y_pred = np.dot(X.values, weights) + bias
    soft_X = softmax(y_pred)
    max = np.argmax(soft_X, axis = 1)
    return np.vectorize(lambda x: x+1)(max)
    
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
    ratings_df = ud['rating']
    genre_df = ud.drop(["movieId","rating","beta0"],axis=1)
    #genre_ud['beta0'] = 1
    ratings_df = ratings_df.astype(int)
    print(ratings_df)
   # print(genre_ud)
    weights, bias, m_error = fit(genre_df, ratings_df, 5, 20000)
    #print(weights, bias, mse)
    pred_y = predict(genre_df, weights, bias)
    print(pred_y)
    error = mse(pred_y, ratings_df)
    print(error)
    """
    a1 = [[ 1.5, 2],[0.2, 3]]
    a2 = [[ 0, 1],[1, 0]]
    s_a1 = softmax(a1)
    a2 = np.array(a2)
    s_a1 = np.array(s_a1)
    print(s_a1 - a2) 
    coefficients, bias = fit(genre_ud, ud_ratings)
    print(coefficients)
    y_pred = np.dot(genre_ud, coefficients) + bias
    
    msee = mse(y_pred,ud_ratings)
    print(msee)
    """
