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

PATH_TO_RATINGS='../../data_movilens/ml-latest-small/ratings.csv'
PATH_TO_MOVIES = '../../data_movilens/ml-latest-small/movies.csv'
PATH_TO_SVD= '../collaborative-filtering/results/model_based_svd/model_svd_0.pickle'
PATH_TO_LINEAR_REGRESSION= '../contet-based/linear_regression0.pickle'
NAME_CROSS_VALIDATION= 'cross_validation_parts.pickle'

def fit(X, y,iter):
    weights = np.full(len(X),1/len(X)) 
    l_step = 0.001
    last_loss = 1000
    for iteration in range(iter):
        y_pred = 0
        for i,ratings in enumerate(X):
            y_pred += np.dot(ratings,weights[i])
        print(y_pred)
        print(y)
        print(np.subtract(y_pred,y))
        print(np.sign(np.subtract(y_pred,y)))
        partial_error = np.sign(np.subtract(y_pred, y))
        for i in range(len(weights)):
            y_pred_i = np.dot(X[i],weights[i])
            nominator = np.dot(partial_error ,y_pred_i)
            partial_derrivation = np.dot(nominator, 1/len(y))
            weights[i] = weights[i] - l_step * partial_derrivation
        loss =   np.sum(np.subtract(y_pred, y)**2) / (abs(len(y)))
        if abs(last_loss - loss) < 0.00000001:
            print(iteration)
            print(abs(last_loss - loss))
            return weights
        print(abs(last_loss - loss))
        last_loss = loss
    return weights
            
        
if __name__ == '__main__':
    dataframe=pd.read_csv(PATH_TO_RATINGS,delimiter=',')
    rating_matrix= pd.pivot_table(data=dataframe,index="userId",columns="movieId", values="rating")

    parts = []
    linear_regression = []
    svd = []

     
    if not os.path.exists("/."+str(NAME_CROSS_VALIDATION)):
        parts = cross_validation.create_parts_dataset(5,131,rating_matrix)
        with open(NAME_CROSS_VALIDATION,"wb") as file:
            pickle.dump(parts,file)
    else:
        with open(NAME_CROSS_VALIDATION,"rb") as file:
            parts = pickle.load(file)

    if not os.path.exists(PATH_TO_SVD) or not os.path.exists(PATH_TO_LINEAR_REGRESSION):
        raise OSError('one or both of the paths to results of algorithms dont exists')
    else: 
        with open(PATH_TO_SVD,'rb') as file:
            svd = pickle.load(file)
        with open(PATH_TO_LINEAR_REGRESSION,'rb') as file:
            linear_regression= pickle.load(file)
    
    svd_arr = [] 
    linear_arr = [] 
    ground_truth = []
     #(607, 1128, 2.5), (607, 4238, 4.0), (607, 46, 4.5), (607, 276, 3.0),
    for rating_tuple in parts[0]:
        row,column,rating=rating_tuple 
        ground_truth.append(rating)
        svd_arr.append(svd.iloc[row,column])
        linear_arr.append(linear_regression.iloc[row,column])
    ground_truth_series = pd.Series(ground_truth)
    svd_series = pd.Series(svd_arr)
    linear_series= pd.Series(linear_arr)
    X = [svd_series, linear_series]
    weights = fit(X,ground_truth_series,1)
    print(weights)
    
    
     

