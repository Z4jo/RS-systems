import sys
sys.path.append('../collaborative-filtering/')
import neighbour_methods
import sim_func
import pandas as pd
import numpy as np
sys.path.append('../math_functions/')

PATH_TO_DATA='../../data_movilens/ml-latest-small/ratings.csv'

def predict_ratings_item_based(item_index,column_vectors,rating_matrix,user_index):
    similarities = []
    top_k_sim = -1  
    all_item_intersections= neighbour_methods.get_intersection(column_vectors,item_index)
    intersection_rating = neighbour_methods.get_ratings_by_item(rating_matrix,all_item_intersections,item_index)
    for rating in intersection_rating:
        similarities.append((sim_func.adjusted_cosine(rating[0],rating[1]),rating[2])) 
    similarities = sorted(similarities, key = lambda x: x[0], reverse = True)
    numerator = 0
    sim = 0  
    top_k = neighbour_methods.get_top_k_users(similarities,top_k_sim,user_index,column_vectors)
    if len(top_k) == 0:
        return None
    for similarity in top_k:
            numerator+=similarity[0]*rating_matrix.iloc[user_index, similarity[1]]
            sim+=abs(similarity[0])
    if sim == 0:
        return None
    outcome = numerator / sim 
    if outcome > 5:
        outcome = 5
    elif outcome < 1:
        outcome = 1

    return outcome 

def main(user_id :int):
    dataframe=pd.read_csv(PATH_TO_DATA,delimiter=',')
    rating_matrix= pd.pivot_table(data=dataframe,index="userId",columns="movieId", values="rating")
    rating_matrix = rating_matrix.reset_index(drop = True)
    rating_matrix.index.name = "userId"
    numbers_array = [num for num in range(0, rating_matrix.shape[1])]
    rating_matrix.columns = numbers_array
    row = rating_matrix.iloc[user_id]
    missing_item_indices = [np.where(row.isnull())[0]]
    column_vectors_item =  neighbour_methods.get_indexes_of_not_empty_ratings_by_item(rating_matrix)
    ret = pd.Series([np.nan]*rating_matrix.shape[1])
    tmp = 0
    for missing_item_index in missing_item_indices[0]: 
         print(tmp)
         item_prediction = predict_ratings_item_based(missing_item_index,  column_vectors_item, rating_matrix,user_id)
         print(item_prediction)
         ret.iloc[missing_item_index]=item_prediction
         tmp+=1

    return ret

if __name__ == '__main__':
    ret = main(0)    
