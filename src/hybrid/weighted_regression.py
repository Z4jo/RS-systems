import numpy as np 
import pandas as pd
import pickle
import os
import sys 
sys.path.append('../cross_validation/')
sys.path.append('../collaborative-filtering/')
sys.path.append('../contet-based/')
import cross_validation
import svd
import linear_regression
import mnl_regression

PATH_TO_RATINGS='../../data_movilens/ml-latest-small/ratings.csv'
PATH_TO_MOVIES = '../../data_movilens/ml-latest-small/movies.csv'
PATH_TO_SVD= '../collaborative-filtering/results/model_based_svd/model_svd_0.pickle'
PATH_TO_LINEAR_REGRESSION= '../contet-based/linear_model.pickle'
PATH_TO_LOGISTIC_REGRESSION= '../contet-based/mnl_model.pickle'
PATH_TO_MODEL = '../hybrid/weighted_model.pickle'
PATH_TO_CROSS= '../cross_validation_parts.pickle'

def fit(X, y,iter):
    weights = np.full(len(X),1/len(X)) 
    l_step = 0.001
    last_loss = 1000
    for iteration in range(iter):
        y_pred = 0
        for i,ratings in enumerate(X):
            y_pred += np.dot(ratings,weights[i])
        partial_error = np.sign(np.subtract(y_pred, y))
        for i in range(len(weights)):
            y_pred_i = np.dot(X[i],weights[i])
            nominator = np.dot(partial_error ,y_pred_i)
            partial_derrivation = np.dot(nominator, 1/len(y))
            weights[i] = weights[i] - l_step * partial_derrivation
        loss =   np.sum(np.subtract(y_pred, y)**2) / (abs(len(y)))
        if abs(last_loss - loss) < 1e-9:
            print(iteration)
            print(abs(last_loss - loss))
            return weights
        last_loss = loss
    return weights

def get_nan_indexes(df):
    not_nan_indexes = []
    nan_indexes = []
    for row_idx, row in enumerate(df.index):
        for col_idx, col in enumerate(df.columns):
            if pd.isna(df.loc[row, col]):
                nan_indexes.append((row_idx,col_idx))
            else:
                not_nan_indexes.append((row_idx, col_idx))

    return nan_indexes,not_nan_indexes

def create_X(rating_matrix,mnl_regression_y_pred, linear_regression_y_pred):
    rating_matrix_clone = rating_matrix.copy()
    numbers_array = [num for num in range(0, rating_matrix_clone.shape[1])]
    rating_matrix_clone.columns = numbers_array
    
    coordinates = []
    mnl_data = []
    linear_data = []

    for i,_ in rating_matrix_clone.iterrows():
        nan_indexes = rating_matrix_clone.iloc[i].index[rating_matrix_clone.iloc[i].isna()]
        mnl_data.append(mnl_regression_y_pred.iloc[i,nan_indexes])
        linear_data.append(linear_regression_y_pred.iloc[i,nan_indexes])
        coordinates.append((i,nan_indexes))
    
    mnl_series = pd.Series(mnl_data)
    linear_series = pd.Series(linear_data)
    X = [linear_series,mnl_series]
    return X,coordinates
    
     
       
def predict(X, weights, coordinates, rating_matrix):
    y_pred = 0
    rating_matrix_clone = rating_matrix.copy()
    numbers_array = [num for num in range(0, rating_matrix_clone.shape[1])]
    rating_matrix_clone.columns = numbers_array
    result_matrix = pd.DataFrame(np.nan, index = range(0,rating_matrix_clone.shape[0]), columns = numbers_array)
    for i,ratings in enumerate(X):
        y_pred += np.dot(ratings,weights[i])
    for i,y in enumerate(y_pred):
        result_matrix.iloc[coordinates[i][0],coordinates[i][1]] = y
    return result_matrix

def predict_unclean(X, weights, coordinates, rating_matrix):
    y_pred = 0
    rating_matrix_clone = rating_matrix.copy()
    numbers_array = [num for num in range(0, rating_matrix_clone.shape[1])]
    rating_matrix_clone.columns = numbers_array
    for i,ratings in enumerate(X):
        y_pred += np.dot(ratings,weights[i])
    for i,y in enumerate(y_pred):
        rating_matrix_clone.iloc[coordinates[i][0],coordinates[i][1]] = y
    return rating_matrix_clone


if __name__ == '__main__':
    dataframe=pd.read_csv(PATH_TO_RATINGS,delimiter=',')
    rating_matrix= pd.pivot_table(data=dataframe,index="userId",columns="movieId", values="rating")
    movies_df = pd.read_csv(PATH_TO_MOVIES,delimiter = ',')
    parts = []
    linear_regression_model = 0
    mnl_regression_model = 0
    
    
    if not os.path.exists("/."+str(PATH_TO_CROSS)):
        parts = cross_validation.create_parts_dataset(5,131,rating_matrix)
        with open(PATH_TO_CROSS,"wb") as file:
            pickle.dump(parts,file)
    else:
        with open(PATH_TO_CROSS,"rb") as file:
            parts = pickle.load(file)

    if not os.path.exists(PATH_TO_LOGISTIC_REGRESSION) or not os.path.exists(PATH_TO_LINEAR_REGRESSION):
        raise OSError('some paht are missing or wrong ')
    else: 
        with open(PATH_TO_LINEAR_REGRESSION,'rb') as file:
            linear_regression_model = pickle.load(file)
        with open(PATH_TO_LOGISTIC_REGRESSION,'rb') as file:
            mnl_regression_model = pickle.load(file)
    


    rating_matrix_clone = rating_matrix.copy()
    for rating_tuple in parts[0]:
        row,column,rating = rating_tuple
        rating_matrix_clone.iloc[row,column] = np.nan

    rating_matrix_clone = rating_matrix_clone.reset_index(drop = True)
    rating_matrix_clone.index.name = "userId"

    all_users_profiles = linear_regression.get_all_users_dataframe(rating_matrix_clone,movies_df)

    linear_regression_cmp = linear_regression.user_profile_predictions(all_users_profiles,linear_regression_model,rating_matrix_clone)
    mnl_regression_cmp = mnl_regression.user_profile_predictions(all_users_profiles,  mnl_regression_model, rating_matrix_clone)

    mnl_arr = []
    linear_arr = []
    ground_truth = []
    weights = [] 


    for rating_tuple in parts[0]:
        row,column,rating=rating_tuple 
        ground_truth.append(rating)
        if pd.isna(mnl_regression_cmp.iloc[row,column])  or pd.isna(linear_regression_cmp.iloc[row,column]) :
            raise AssertionError('nan as value')
        mnl_arr.append(mnl_regression_cmp.iloc[row,column])
        linear_arr.append(linear_regression_cmp.iloc[row,column])
    ground_truth_series = pd.Series(ground_truth)
    mnl_series = pd.Series(mnl_arr)
    linear_series= pd.Series(linear_arr)
    
    X = [linear_series, mnl_series]
    model = []
    if os.path.exists(PATH_TO_MODEL):
        with open(PATH_TO_MODEL,'rb') as file:
            model = pickle.load(file)
    else:
        model = fit(X,ground_truth_series,100000)
        with open('../hybrid/weighted_model.pickle','wb') as file:
            pickle.dump(model,file)

    for iteration,part in enumerate(parts):
        rating_matrix_clone = rating_matrix.copy()
        for rating_tuple in part:
            row,column,rating = rating_tuple 
            rating_matrix_clone.iloc[row,column] = np.nan

        rating_matrix_clone = rating_matrix_clone.reset_index(drop = True)
        rating_matrix_clone.index.name = "userId"

        all_users_dataframes = mnl_regression.get_all_users_dataframe(rating_matrix_clone.copy(),movies_df)
        mnl_regression_cmp = mnl_regression.user_profile_predictions(all_users_dataframes,mnl_regression_model,rating_matrix_clone)
        linear_regression_cmp = linear_regression.user_profile_predictions(all_users_dataframes,linear_regression_model,rating_matrix_clone) 

        X, coordinates = create_X(rating_matrix_clone,mnl_regression_cmp,linear_regression_cmp)
        predictions = predict(X, model, coordinates, rating_matrix_clone)

        with open('../hybrid/weighted_regression'+str(iteration)+'.pickle','wb') as file:
            pickle.dump(predictions,file)
        print(f"iteration end:{iteration}") 
     
