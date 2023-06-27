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
        reconstructed_matrix = np.dot(np.dot(U,sig),V)
        reconstructed_df = pd.DataFrame(reconstructed_matrix)
        print(f"recoM:{reconstructed_df}")
#        with open("model_svd_"+str(iteration)+".pickle","wb") as file:
#            pickle.dump(reconstructed_df,file)
