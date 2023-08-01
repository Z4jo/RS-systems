import numpy as np 
import pandas as pd
import pickle
import os
import sys 
sys.path.append('../cross_validation/')
sys.path.append('../collaborative-filtering/')
sys.path.append('../contet-based/')
sys.path.append('../hybrid/')
import cross_validation
import svd
import svd_demo
import weighted_regression
import linear_regression
import mnl_regression

PATH_TO_RATINGS = '../../data_movilens/ml-latest-small/ratings.csv'
PATH_TO_MOVIES = '../../data_movilens/ml-latest-small/movies.csv'
PATH_TO_SVD = '../collaborative-filtering/results/model_based_svd/model_svd_0.pickle'
PATH_TO_WEIGHTED_MODEL = '../hybrid/weighted_model.pickle'
PATH_TO_LINEAR_REGRESSION_MODEL = '../contet-based/linear_model.pickle'
PATH_TO_LOGISTIC_REGRESSION_MODEL = '../contet-based/mnl_model.pickle'
PATH_TO_CROSS= '../cross_validation_parts.pickle'

 
if __name__ == '__main__':
    ratings_df = pd.read_csv(PATH_TO_RATINGS,delimiter = ',')
    movies_df = pd.read_csv(PATH_TO_MOVIES,delimiter = ',')
    rating_matrix= pd.pivot_table(data=ratings_df,index="userId",columns="movieId", values="rating")


    parts = []
    if not os.path.exists("./"+str(PATH_TO_CROSS)):
        parts = cross_validation.create_parts_dataset(5,131,rating_matrix)
        with open(PATH_TO_CROSS,"wb") as file:
            pickle.dump(parts,file)
    else:
        with open(PATH_TO_CROSS,"rb") as file:
            parts = pickle.load(file)

    model = []
    linear_regression_model = []
    mnl_regression_model = []

    if not os.path.exists(PATH_TO_WEIGHTED_MODEL): 
        raise OSError("one of the paths to the models doesn't exist")
    else: 
        with open (PATH_TO_WEIGHTED_MODEL,'rb') as file:
            model = pickle.load(file)
        with open(PATH_TO_LINEAR_REGRESSION_MODEL,'rb') as file:
            linear_regression_model = pickle.load(file)
        with open(PATH_TO_LOGISTIC_REGRESSION_MODEL,'rb') as file:
            mnl_regression_model = pickle.load(file)
    
    
        
    for iteration, part in enumerate(parts):
        rating_matrix_clone = rating_matrix.copy()

        for rating_tuple in part:
            row,column,rating = rating_tuple
            rating_matrix_clone.iloc[row,column] = np.nan

        rating_matrix_clone = rating_matrix_clone.reset_index(drop = True)
        rating_matrix_clone.index.name = "userId"

        all_users_dataframes = mnl_regression.get_all_users_dataframe(rating_matrix_clone.copy(),movies_df)
        mnl_regression_cmp = mnl_regression.user_profile_predictions(all_users_dataframes,mnl_regression_model,rating_matrix_clone)
        linear_regression_cmp = linear_regression.user_profile_predictions(all_users_dataframes,linear_regression_model,rating_matrix_clone) 
        X, coordinates = weighted_regression.create_X(rating_matrix_clone,mnl_regression_cmp,linear_regression_cmp)
        y_pred = weighted_regression.predict_unclean(X,model,coordinates,rating_matrix_clone)
        reconstructed_ratings = svd.predict(y_pred)
        final_df = svd_demo.clear_result(user_id,reconstructed_ratings,rating_matrix_clone)
        

        print("final")
        print(final_df)

        with open('../hybrid/cascande_weighted_svd_2_'+str(iteration)+'.pickle','wb') as file:
            pickle.dump(final_df,file)
        
        print(f"iteration done:{iteration}")
     


