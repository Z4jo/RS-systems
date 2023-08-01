import typer
import pandas as pd
import mnl_regression_demo
import linear_regression_demo
import naive_bayes_demo
import item_based_demo
import user_based_demo
import svd_demo
import weighted_hybrid_demo
import cascade_mnl_svd_demo
import cascade_weighted_svd_demo
import time

PATH_TO_MOVIES = '../../data_movilens/ml-latest-small/movies.csv'
LIST_OF_ALGOS = ['user_based', 'item_based', 'svd', 'linear_regression', 'mnl_regression', 'naive_bayes','weighted_regression','cascade_mnl_svd','cascade_weighted_svd']

def main(name_of_algo: str, user_id: int):
    movies_df = pd.read_csv(PATH_TO_MOVIES,delimiter = ',')
    if name_of_algo not in LIST_OF_ALGOS:
        raise ValueError(f"there is no algorithm like that chose one from this list:{LIST_OF_ALGOS}")
    if user_id < 1 or user_id > 610:
        raise ValueError(f"user_id has to be between 1 and 610")
    
    user_id = user_id - 1 
    ret = 0
    start_time = 0 
    if name_of_algo == LIST_OF_ALGOS[0]:
        start_time = time.time()
        ret = user_based_demo.main(user_id)
    if name_of_algo == LIST_OF_ALGOS[1]:
        raise Exception('todo')
        #ret = item_based_demo.main(user_id)
    if name_of_algo == LIST_OF_ALGOS[2]:
        start_time = time.time()
        ret = svd_demo.main(user_id)
    if name_of_algo == LIST_OF_ALGOS[3]:
        start_time = time.time()
        ret = linear_regression_demo.main(user_id)
    if name_of_algo == LIST_OF_ALGOS[4]:
        start_time = time.time()
        ret = mnl_regression_demo.main(user_id)
    if name_of_algo == LIST_OF_ALGOS[5]:
        start_time = time.time()
        ret = naive_bayes_demo.main(user_id)
    if name_of_algo == LIST_OF_ALGOS[6]:
        start_time = time.time()
        ret = weighted_hybrid_demo.main(user_id)
    if name_of_algo == LIST_OF_ALGOS[7]:
        start_time = time.time()
        ret = cascade_mnl_svd_demo.main(user_id)
    if name_of_algo == LIST_OF_ALGOS[8]:
        start_time = time.time()
        ret = cascade_weighted_svd_demo.main(user_id)
     
    ret = ret.sort_values(ascending = False, na_position = 'last')
    top_20 = ret.iloc[:20]
    
    movies_indeces = top_20.index.values
    movies = movies_df.iloc[movies_indeces] 
    movies = movies.assign(predicted_ratings = top_20)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(movies)
    print(f"time for recommendation:{elapsed_time}")

if __name__ == '__main__':
    typer.run(main)
    
