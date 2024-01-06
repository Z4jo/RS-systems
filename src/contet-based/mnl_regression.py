import numpy as np 
import pandas as pd
import math 
import multiprocessing
import pickle 
import sys 
import os
sys.path.append('../cross_validation/')
import cross_validation
PATH_TO_RATINGS='../../data_movilens/ml-latest-small/ratings.csv'
PATH_TO_MOVIES = '../../data_movilens/ml-latest-small/movies.csv'
PATH_TO_MODEL = '../contet-based/mnl_model.pickle'
PATH_TO_CROSS = '../cross_validation_parts.pickle'


def generate_user_dataframe(rating_matrix, movies_df, user_id):
    user_ratings_df = rating_matrix.iloc[user_id]
    user_ratings_df = user_ratings_df.to_frame('rating').reset_index()
    movie_id_indeces = user_ratings_df['movieId']
    filtered_movies_df = movies_df[movies_df['movieId'].isin(movie_id_indeces)]
    genres_df = filtered_movies_df['genres'].str.get_dummies('|')
    genres_id = pd.concat([filtered_movies_df['movieId'],genres_df], axis = 1)
    merged_df = pd.merge(genres_id, user_ratings_df, on='movieId', how='left')
    return merged_df

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def mse(pred_y, y):
    total = 0
    for i in range(len(pred_y)):
        total += (pred_y[i]-y.iloc[i])**2
    return (1/len(pred_y))*total

def onehot(ratings, num_classes):
    encoded = np.zeros((len(ratings), num_classes))
    ratings_adjusted = ratings - 1
    encoded[np.arange(len(ratings)), ratings_adjusted] = 1
    return encoded

def fit(X, y,class_number, iter):
    weights = np.zeros((X.shape[1],class_number))
    bias = np.zeros((class_number))
    l_rate = 0.001
    last_loss = 1000
    for i in range(iter):
        y_pred = np.dot(X,weights) + bias
        dw = (1/X.shape[0])*np.dot(X.values.T,softmax(y_pred) - onehot(y, class_number))
        db = (1/X.shape[0])*np.sum(softmax(y_pred) - onehot(y, class_number))
        weights = weights - l_rate * dw
        bias = bias - l_rate * db
        loss = -np.mean(np.log(softmax(y_pred)[np.arange(len(y)), np.vectorize(lambda x: x - 1)(y)]))
        if abs(last_loss - loss) < 1e-6:
            print(f"iter:{i}")
            return weights, bias 
        last_loss = loss
    return weights, bias 

def calculate_prediction(X, weights, bias):
    y_pred = np.dot(X.values, weights) + bias
    soft_X = softmax(y_pred)
    max = np.argmax(soft_X, axis = 1)
    return np.vectorize(lambda x: x+1)(max)

def get_all_users_dataframe(rating_matrix, movies_df):
    all_users_dataframes = []
    for i,_ in rating_matrix.iterrows():
        user_df = generate_user_dataframe(rating_matrix,movies_df,i)
        all_users_dataframes.append((user_df, i))
    return all_users_dataframes

def create_model(all_users_dataframes):
    print('creating model')
    model = []
    pool = multiprocessing.Pool(processes=8)
    user_profiles = pool.starmap(model_generation, all_users_dataframes)
    pool.close()
    pool.join()
    for _,weights, bias in user_profiles:
        model.append((weights,bias))
    with open(PATH_TO_MODEL,"wb") as file:
        pickle.dump(model,file)
    return model

def model_generation(ud,index):
    ud = ud.rename(columns = {"(no genres listed)": "beta0" })
    ud = ud.dropna()
    genre_df = ud.drop(["movieId","rating","beta0"],axis=1)
    ratings_df = ud['rating']
    ratings_df = ratings_df.astype(int)
    weights, bias = fit(genre_df, ratings_df, 5, 50000)
    return (index,weights,bias)


def user_profile_predictions_unlcean(all_users_dataframes,model,rating_matrix):
    rating_matrix_clone = rating_matrix.copy()
    numbers_array = [num for num in range(0, rating_matrix_clone.shape[1])]
    rating_matrix_clone.columns = numbers_array
    for ud,i in all_users_dataframes:
        ud = ud.rename(columns = {"(no genres listed)": "beta0" })
        nan_df = ud[ud['rating'].isna()].drop(["movieId","rating","beta0"],axis=1)
        weights, bias = model[i]
        y_pred = calculate_prediction(nan_df, weights, bias)
        nan_indexes = rating_matrix_clone.iloc[i].index[rating_matrix_clone.iloc[i].isna()] 
        if y_pred.shape == nan_indexes.shape:
            rating_matrix_clone.iloc[i,nan_indexes] = y_pred
    return rating_matrix_clone

def user_profile_predictions(all_users_dataframes,model,rating_matrix):
    rating_matrix_clone = rating_matrix.copy()
    numbers_array = [num for num in range(0, rating_matrix_clone.shape[1])]
    rating_matrix_clone.columns = numbers_array
    result_matrix = pd.DataFrame(np.nan,index = range(0,rating_matrix_clone.shape[0]),columns = numbers_array)
    for ud,i in all_users_dataframes:
        ud = ud.rename(columns = {"(no genres listed)": "beta0" })
        nan_df = ud[ud['rating'].isna()].drop(["movieId","rating","beta0"],axis=1)
        weights, bias = model[i]
        y_pred = calculate_prediction(nan_df, weights, bias)
        nan_indexes = rating_matrix_clone.iloc[i].index[rating_matrix_clone.iloc[i].isna()] 
        if y_pred.shape == nan_indexes.shape:
            result_matrix.iloc[i,nan_indexes] = y_pred
    return result_matrix

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
    if not os.path.exists(PATH_TO_CROSS):
        parts = cross_validation.create_parts_dataset(5,131,rating_matrix)
        with open(PATH_TO_CROSS,"wb") as file:
            pickle.dump(parts,file)
    else:
        with open(PATH_TO_CROSS,"rb") as file:
            parts = pickle.load(file)

    rating_matrix = rating_matrix.reset_index(drop = True)
    rating_matrix.index.name = "userId"
    rating_matrix_clone = rating_matrix.copy()

    for rating_tuple in parts[0]:
        row,column,rating=rating_tuple
        rating_matrix_clone.iloc[row,column] = np.nan

    all_users_dataframes = get_all_users_dataframe(rating_matrix_clone,movies_df)
    model = []
    if (os.path.exists(PATH_TO_MODEL)):
        with open(PATH_TO_MODEL,"rb") as file:
            model = pickle.load(file)
    else:
        model = create_model(all_users_dataframes)

    for iteration,part in enumerate(parts):
        rating_matrix_clone = rating_matrix.copy()
        for rating_tuple in part:
            row,column,rating=rating_tuple
            rating_matrix_clone.iloc[row,column] = np.nan

        rating_matrix_clone = rating_matrix_clone.reset_index(drop = True)
        rating_matrix_clone.index.name = "userId"

        all_users_dataframes = get_all_users_dataframe(rating_matrix_clone.copy(),movies_df)
       
        result = user_profile_predictions(all_users_dataframes,model,rating_matrix_clone)

        with open('../contet-based/mnl_regression'+str(iteration)+'.pickle', 'wb') as file:
             pickle.dump(result,file)
        print(f"iteration done:{iteration}")

