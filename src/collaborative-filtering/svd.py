import numpy as np
import pandas as pd
import os
import pickle
import sys
sys.path.append('../cross_validation/')
import cross_validation

#NOTE: MOVIELENS DATASET
PATH_TO_DATA='../../data_movilens/ml-latest-small/ratings.csv'
#NOTE: TEST DATASET
#PATH_TO_DATA='../../data_movilens/testSVD.csv'

def get_not_nan_indexes(df):
    not_nan_indexes = []

    for row_idx, row in enumerate(df.index):
        for col_idx, col in enumerate(df.columns):
            if not pd.isna(df.loc[row, col]):
                not_nan_indexes.append((row_idx, col_idx))

    return not_nan_indexes

if __name__ == '__main__':
    dataframe=pd.read_csv(PATH_TO_DATA,delimiter=',')
    rating_matrix= pd.pivot_table(data=dataframe,index="userId",columns="movieId", values="rating")
    print(rating_matrix)

    parts = []
   
    if not os.path.exists("./cross_validation_parts.pickle"):
        parts = cross_validation.create_parts_dataset(5,131,rating_matrix)
        with open("cross_validation_parts.pickle","wb") as file:
            pickle.dump(parts,file)
    else:
        with open("cross_validation_parts.pickle","rb") as file:
            parts = pickle.load(file)

    for iteration, part in enumerate(parts):
        rating_matrix_clone = rating_matrix.copy()
        for rating_tuple in part:
            row,column,rating=rating_tuple
            rating_matrix_clone.iloc[row,column] = np.nan
        mean = rating_matrix_clone.mean(1)
        filled_rm = rating_matrix.apply(lambda row: row.fillna(mean[row.name]),axis = 1 )
        np_matrix= filled_rm.values
        U, sig, V = np.linalg.svd(np_matrix)
        k = 20
        sig = np.diag(sig)
        sig=sig[0:k,0:k]
        U=U[:,0:k]
        V=V[0:k,:] 
        #print(f"U:{U}\n Sigma:{sig}\n Vt:{V}")

        numbers_array = [num for num in range(0, rating_matrix_clone.shape[1])]
        rating_matrix_clone.columns = numbers_array
        not_nan_indexes = get_not_nan_indexes(rating_matrix_clone)
        reconstructed_matrix = np.dot(np.dot(U,sig),V)
        reconstructed_df = pd.DataFrame(reconstructed_matrix)
        for index in not_nan_indexes:
            reconstructed_df.iloc[index[0], index[1]] = np.nan

        print(f"recoM:{reconstructed_df}")

        with open("model_svd_"+str(iteration)+".pickle","wb") as file:
            pickle.dump(reconstructed_df,file)
