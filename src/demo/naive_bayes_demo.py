import sys
sys.path.append('../contet-based/')
import naive_bayes
import pandas as pd
import numpy as np
import pickle

PATH_TO_RATINGS='../../data_movilens/ml-latest-small/ratings.csv'
PATH_TO_MOVIES = '../../data_movilens/ml-latest-small/movies.csv'
def main(user_id: int):
    ratings_df = pd.read_csv(PATH_TO_RATINGS,delimiter = ',')
    movies_df = pd.read_csv(PATH_TO_MOVIES,delimiter = ',')
    rating_matrix= pd.pivot_table(data=ratings_df,index="userId",columns="movieId", values="rating")

    user_count,item_count = rating_matrix.shape
    ud = naive_bayes.generate_user_dataframe(rating_matrix,movies_df,user_id)

    predictions,_,_= naive_bayes.naive_bayes_prediciton(ud,user_id,0)
    result_dataframe = pd.DataFrame(np.nan, index = range(1), columns = rating_matrix.columns)

    for prediction in predictions:
        movie_id = int(prediction[1])
        result_dataframe.loc[user_id,movie_id] = prediction[0]
        
    column_arr = [i for i in range(item_count)] 
    result_dataframe.columns = column_arr
    result_series = result_dataframe.iloc[0]
    
    return result_series
    

if __name__ == '__main__':

    main(0)
