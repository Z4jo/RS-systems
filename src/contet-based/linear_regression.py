import numpy as np 
import pandas as pd
import pickle 
import os
import sys
import multiprocessing
sys.path.append('../cross_validation/')
import cross_validation

PATH_TO_RATINGS='../../data_movilens/ml-latest-small/ratings.csv'
PATH_TO_MOVIES = '../../data_movilens/ml-latest-small/movies.csv'
PATH_TO_MODEL = '../contet-based/linear_model.pickle'
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

def mse(pred_y, y):
    total = 0
    for i in range(len(pred_y)):
        total += (pred_y[i]-y.iloc[i])**2
    return (1/len(pred_y))*total
        
def fit(X, y, iter):
    weights = np.zeros(X.shape[1])
    bias = 0
    l_rate = 0.01
    last_mse = 1000
    for i in range(iter):
        y_pred = np.dot(X, weights) + bias
        test = mse(y_pred, y)
        if abs(last_mse - test)< 1e-6:
            print("iteration:",i)
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
   

def user_model_creation(ud,index):
    ud = ud.rename(columns = {"(no genres listed)": "beta0" })
    movies_id= ud[ud['rating'].isna()]
    movies_id= movies_id['movieId']
    ud = ud.dropna()
    ud_ratings = ud['rating']
    genre_ud = ud.drop(["movieId","rating","beta0"],axis=1)

    coefficients, bias = fit(genre_ud, ud_ratings, 10000)
    return (index,coefficients,bias)

def get_all_users_dataframe(rating_matrix,movies_df):
    all_users_dataframes = []
    for i, _ in rating_matrix.iterrows():
        user_profile = generate_user_dataframe(rating_matrix,movies_df, i)
        all_users_dataframes.append((user_profile,i))
    return all_users_dataframes

def user_profile_predictions(all_users_dataframes,model,rating_matrix):
    rating_matrix_clone = rating_matrix.copy()
    numbers_array = [num for num in range(0, rating_matrix_clone.shape[1])]
    rating_matrix_clone.columns = numbers_array
    result_matrix = pd.DataFrame(np.nan,index = range(0,rating_matrix_clone.shape[0]),columns = numbers_array)

    for ud, i in all_users_dataframes:
        ud = ud.rename(columns = {"(no genres listed)": "beta0" })
        nan_df= ud[ud['rating'].isna()].drop(["movieId","rating","beta0"],axis=1)
        coefficients, bias = model[i]
        y_pred = np.dot(nan_df, coefficients) + bias
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

def create_model(all_users_dataframes):
    print('creating model')
    pool = multiprocessing.Pool(processes=8)
    users_models= pool.starmap(user_model_creation, all_users_dataframes)
    pool.close()
    pool.join()
    model = []
    for _, coefficients, bias in users_models:
        model.append((coefficients,bias))
    with open("linear_model_to_delete.pickle","wb") as file:
        pickle.dump(model,file)
    return model

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
    if os.path.exists(PATH_TO_MODEL):
        with open(PATH_TO_MODEL,"rb")as file:
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

        with open('../contet-based/linear_regression'+str(iteration)+'.pickle', 'wb') as file:
            pickle.dump(result,file)
        print(f"iteration done:{iteration}")
            
    
