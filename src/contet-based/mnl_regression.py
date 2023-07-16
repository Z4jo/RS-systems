import numpy as np 
import pandas as pd
import math 
import multiprocessing
import pickle 
import sys 
import os
sys.path.append('../cross_validation/')
import cross_validation
#PATH_TO_MOVIES = '../../data_movilens/content-based movies.csv'
#PATH_TO_RATINGS = '../../data_movilens/content-based ratings.csv'
PATH_TO_RATINGS='../../data_movilens/ml-latest-small/ratings.csv'
PATH_TO_MOVIES = '../../data_movilens/ml-latest-small/movies.csv'
NAME_CROSS_VALIDATION= 'cross_validation_parts.pickle'


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
    last_loss = 1000
    print(weights.shape)
    print(X.shape)
    for i in range(iter):
        y_pred = np.dot(X,weights) + bias
        dw = (1/X.shape[0])*np.dot(X.values.T,softmax(y_pred) - onehot(y, class_number))
        db = (1/X.shape[0])*np.sum(softmax(y_pred) - onehot(y, class_number))
        weights = weights - l_rate * dw
        bias = bias - l_rate * db
        loss = -np.mean(np.log(softmax(y_pred)[np.arange(len(y)), np.vectorize(lambda x: x - 1)(y)]))
        if abs(last_loss - loss)< 0.0000001: 
            print(i)
            return weights, bias 

    return weights, bias 

def calculate_prediction(X, weights, bias):
    y_pred = np.dot(X.values, weights) + bias
    soft_X = softmax(y_pred)
    max = np.argmax(soft_X, axis = 1)
    return np.vectorize(lambda x: x+1)(max)
    
def user_profile_prediction(ud,index):
    ud = ud.dropna()
    ud = ud.rename(columns = {"(no genres listed)": "beta0" })
    ratings_df = ud['rating']
    genre_df = ud.drop(["movieId","rating","beta0"],axis=1)
    ratings_df = ratings_df.astype(int)
    weights, bias = fit(genre_df, ratings_df, 5, 20000)
    pred_y = calculate_prediction(genre_df, weights, bias)
    return (pred_y,index)

def predict(rating_matrix, movies_df):
    rating_matrix = rating_matrix.reset_index(drop = True)
    rating_matrix.index.name = "userId"
    iterable = []
    for i, _ in rating_matrix.iterrows():
        user_data = generate_user_dataframe(rating_matrix,movies_df,i)
        iterable.append((user_data,i))

    pool = multiprocessing.Pool(processes=6)
    users_predictions = pool.starmap(user_profile_prediction, iterable)
    pool.close()
    pool.join()
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

def get_not_nan_indexes(df):
    not_nan_indexes = []

    for row_idx, row in enumerate(df.index):
        for col_idx, col in enumerate(df.columns):
            if not pd.isna(df.loc[row, col]):
                not_nan_indexes.append((row_idx, col_idx))

    return not_nan_indexes

if __name__ == '__main__':
    ratings_df = pd.read_csv(PATH_TO_RATINGS,delimiter = ',')
    movies_df = pd.read_csv(PATH_TO_MOVIES,delimiter = ',')
    rating_matrix= pd.pivot_table(data=ratings_df,index="userId",columns="movieId", values="rating")
    parts = []
    if not os.path.exists("./"+str(NAME_CROSS_VALIDATION)):
        parts = cross_validation.create_parts_dataset(5,131,rating_matrix)
        with open(NAME_CROSS_VALIDATION,"wb") as file:
            pickle.dump(parts,file)
    else:
        with open(NAME_CROSS_VALIDATION,"rb") as file:
            parts = pickle.load(file)

    for iteration,part in enumerate(parts):
        rating_matrix_clone = rating_matrix.copy()
        for rating_tuple in part:
            row,column,rating=rating_tuple
            rating_matrix_clone.iloc[row,column] = np.nan
        result = predict(rating_matrix_clone,movies_df)

        numbers_array = [num for num in range(0, rating_matrix.shape[1])]
        rating_matrix_clone.columns = numbers_array
        not_nan_indexes = get_not_nan_indexes(rating_matrix_clone)
        
        for index in not_nan_indexes:
            result.iloc[index[0], index[1]] = np.nan


        with open('../contet-based/mnl_regression'+str(iteration)+'.pickle', 'wb') as file:
            pickle.dump(result,file)
        print(f"iteration done:{iteration}")
            

