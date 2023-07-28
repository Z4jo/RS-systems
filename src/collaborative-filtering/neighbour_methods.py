import numpy as np
import pandas as pd
import pickle
import sys
import os
import multiprocessing 
import cProfile
import pstats
sys.path.append('../cross_validation/')
sys.path.append('../data_procession/')
sys.path.append('../math_functions/')
import sim_func
import cross_validation

#PATH_TO_DATA='../../data_movilens/test.csv'
PATH_TO_DATA='../../data_movilens/ml-latest-small/ratings.csv'

def get_ratings_by_user(rating_matrix,intersections,index_active_user):
    users_ratings= []
    for intersection in intersections:
        ratings_neighbour=[]
        ratings_active_user=[]
        for column in intersection[0]:
            ratings_neighbour.append(rating_matrix.iloc[intersection[1],column])
            ratings_active_user.append(rating_matrix.iloc[index_active_user,column])
        users_ratings.append((ratings_neighbour,ratings_active_user,intersection[1]))
    return users_ratings        

def get_ratings_by_item(rating_matrix,intersections,index_active_movie):
    rating_matrix = mean_center_matrix(rating_matrix)
    movies_ratings= []
    for intersection in intersections:
        ratings_neighbour=[]
        ratings_active_movie=[]
        for row in intersection[0]:
            ratings_neighbour.append(rating_matrix.iloc[row,intersection[1]])
            ratings_active_movie.append(rating_matrix.iloc[row,index_active_movie])
        movies_ratings.append((ratings_neighbour,ratings_active_movie,intersection[1]))
    return movies_ratings 

def mean_center_matrix(rating_matrix): 
    mean = rating_matrix.mean(1)
    mean_centered_rm = rating_matrix.sub(mean, axis=0)
    return mean_centered_rm 

def get_intersection(vectors, active_index):
    intersections = []
    for i,not_null_indexes in enumerate(vectors):
            intersection = (np.intersect1d(np.array(vectors[active_index]),np.array(not_null_indexes)),i)
            if len(intersection[0])>5:
                intersections.append(intersection)
    return intersections 

def get_indexes_of_not_empty_ratings_by_item(rating_matrix):
    not_emnpty_column_vectors = []
    for column,_ in enumerate(rating_matrix.columns): 
        not_empty_indexes=[] 
        for row,_ in enumerate(rating_matrix.index):
            cell_value=rating_matrix.iloc[row,column]
            if not np.isnan(cell_value):
                not_empty_indexes.append(row) 
        not_emnpty_column_vectors.append(not_empty_indexes)
    return not_emnpty_column_vectors 

def get_indexes_of_not_empty_ratings_by_user(df):
    all_not_nan_indeces = []
    for _,row in df.iterrows():
        non_nan_indexes = [col for col, value in enumerate(row.values) if pd.notnull(value)]
        all_not_nan_indeces.append(non_nan_indexes)
    return all_not_nan_indeces

def get_indexes_of_missing_ratings_by_item(column_index,rating_matrix):
    missing_indexes=[]
    for row,_ in enumerate(rating_matrix.index):
        cell_value = rating_matrix.iloc[row,column_index]
        if np.isnan(cell_value):
           missing_indexes.append((row,column_index))
    return missing_indexes

def get_indexes_of_not_empty_columns(rating_matrix):
    not_emnpty_column_vectors = []
    for row in range(len(rating_matrix)): 
        not_empty_indexes=[] 
        for column,_ in enumerate(rating_matrix.columns):
            cell_value=rating_matrix.iloc[row,column]
            if not np.isnan(cell_value):
                not_empty_indexes.append(column) 
        not_emnpty_column_vectors.append(not_empty_indexes)
    return not_emnpty_column_vectors    

def get_top_k_users(soreted_similarities,size,missing_rating_index,column_vectors):
    ret = []
    if size > len(soreted_similarities) and size > -1:
        return [] 
    for similarity in soreted_similarities: 
        if len(ret)==size:
         return ret
        if  missing_rating_index in column_vectors[similarity[1]]:
            ret.append((similarity[0],similarity[1]))
    return ret

def predict_ratings_user_based(user_index, missing_entries, column_vectors, rating_matrix):
    top_k_users_size = -1 
    similarities = []
    predictions=[]
    users_mean= rating_matrix.mean(1)
    all_user_intersections = get_intersection(column_vectors,user_index)
    intersection_rating = get_ratings_by_user(rating_matrix,all_user_intersections,user_index)
    for rating in intersection_rating:
        #NOTE: UNCOMENT FOR COSINE SIMILARTIY
        #similarities.append((sim_func.adjusted_cosine(rating[0],rating[1]),rating[2]))
        #NOTE: UNCOMENT FOR PEARSON_COEFICIENT 
        similarities.append((sim_func.pearson_coefficient(rating[0],rating[1]),rating[2]))
    sorted_similarities= sorted(similarities, key=lambda x: x[0],reverse=True)
    for missing_rating_index in missing_entries:
        numerator = 0
        sim = 0 
        top_k_users=get_top_k_users(sorted_similarities,top_k_users_size,missing_rating_index,column_vectors)
        if len(top_k_users)==0:
            continue
        for similarity in top_k_users:
            numerator+=similarity[0]*(rating_matrix.iloc[similarity[1], missing_rating_index]-users_mean.iloc[similarity[1]])
            sim+=similarity[0]
        if sim == 0: 
            continue
        prediction = (user_index,missing_rating_index,users_mean.iloc[user_index]+(numerator/sim))
        predictions.append(prediction)
    return predictions 

def predict_ratings_item_based(item_index,missing_entries,column_vectors,rating_matrix):
    similarities = []
    result_series = pd.Series([np.nan] * 610)
    top_k_sim = -1
    all_item_intersections= get_intersection(column_vectors,item_index)
    intersection_rating = get_ratings_by_item(rating_matrix,all_item_intersections,item_index)
    similarities = sorted(similarities, key = lambda x: x[0], reverse = True)
    for rating in intersection_rating:
        #NOTE:UNCOMENT FOR COSINE SIMILARTIY 
        similarities.append((sim_func.adjusted_cosine(rating[0],rating[1]),rating[2])) 
        #NOTE:UNCOMENT FOR PEARSON_COEFICIENT 
        #similarities.append((sim_func.pearson_coefficient(rating[0],rating[1]),rating[2]))
    for missing_rating_index in missing_entries:
            numerator = 0
            sim = 0  
            top_k = get_top_k_users(similarities,top_k_sim,missing_rating_index,column_vectors)
            if len(top_k) == 0:
                continue
            for similarity in top_k:
                    numerator+=similarity[0]*rating_matrix.iloc[missing_rating_index, similarity[1]]
                    sim+=abs(similarity[0])
            if sim == 0:
                continue
            outcome = numerator / sim 
            result_series.iloc[missing_rating_index] = outcome 
    return result_series
            

def item_based_setup(rating_matrix,number_of_iteration):
    missing_indeces = []
    for column in rating_matrix.columns:
            nan_indexes = np.where(rating_matrix[column].isna())[0]
            missing_indeces.append(nan_indexes)
    user_rating_indexes = 0
    name_of_pickle_file='user_rating_indexes'+str(number_of_iteration)+'.pickle'
    print(name_of_pickle_file)
    file_path = os.path.join('./', name_of_pickle_file)  # Construct the full file path
    if not os.path.exists(file_path):
        user_rating_indexes=get_indexes_of_not_empty_ratings_by_item(rating_matrix)
        with open(name_of_pickle_file, 'wb') as file:
            pickle.dump(user_rating_indexes,file)
    else:
        with open(name_of_pickle_file, 'rb') as file:
            user_rating_indexes=pickle.load(file) 
    iterable = []
    for item_index,missing_entries in enumerate(missing_indeces):
        iterable.append((item_index,missing_entries,user_rating_indexes,rating_matrix))
    pool = multiprocessing.Pool(processes=8)
    series = pool.starmap(predict_ratings_item_based,iterable)
    pool.close()
    pool.join()
    final_dataframe = pd.DataFrame(series)
    return final_dataframe.T

def user_based_setup(rating_matrix, number_of_iteration):
    missing_indices = [np.where(row.isnull())[0] for _, row in rating_matrix.iterrows()]
    item_rating_indexes = 0
    name_of_pickle_file='item_rating_indexes'+str(number_of_iteration)+'.pickle'
    print(name_of_pickle_file)
    file_path = os.path.join('./', name_of_pickle_file)  # Construct the full file path
    if not os.path.exists(file_path):
        item_rating_indexes =  get_indexes_of_not_empty_ratings_by_user(rating_matrix)
        print(item_rating_indexes)
        with open(name_of_pickle_file, 'wb') as file:
            pickle.dump(item_rating_indexes,file)
    else:
        with open(name_of_pickle_file, 'rb') as file:
            item_rating_indexes=pickle.load(file) 
    iterable = []
    for user_index, missing_entries in enumerate(missing_indices):
        iterable.append((user_index, missing_entries ,item_rating_indexes,rating_matrix))
    pool = multiprocessing.Pool(processes=8)
    results = pool.starmap(predict_ratings_user_based,iterable)
    pool.close()
    pool.join()
    return results

if __name__ == '__main__':
    dataframe=pd.read_csv(PATH_TO_DATA,delimiter=',')
    rating_matrix= pd.pivot_table(data=dataframe,index="userId",columns="movieId", values="rating")
    print(rating_matrix.shape)
     
    parts = []
   
    if not os.path.exists("./cross_validation_parts.pickle"):
        parts = cross_validation.create_parts_dataset(5,131,rating_matrix)
        with open("cross_validation_parts.pickle","wb") as file:
            pickle.dump(parts,file)
    else:
        with open("cross_validation_parts.pickle","rb") as file:
            parts = pickle.load(file)
   
    profiler = cProfile.Profile()
    for iteration, part in enumerate(parts):
        rating_matrix_clone = rating_matrix.copy()
        for rating_tuple in part:
            row,column,rating=rating_tuple
            rating_matrix_clone.iloc[row,column] = np.nan
        profiler.enable()
        #result_prediction = user_based_setup(rating_matrix_clone,iteration)
        result = item_based_setup(rating_matrix_clone,iteration)
        profiler.disable()
        profiler.dump_stats('profile_stats')
        # Create a pstats.Stats object
        stats = pstats.Stats('profile_stats')
        # Print the statistics
        stats.print_stats()
        #WARN: depends on the algo
        with open("item_based_results_cosine_test_10_"+str(iteration)+".pickle", 'wb') as file:
            pickle.dump(result,file)
"""
 #NOTE: for tests

    for rating in parts[0]:
        rating_matrix.iloc[rating[0],rating[1]] = np.nan
    
    column_vectors= get_indexes_of_not_empty_ratings_by_user2(rating_matrix)
    missing_indices = [np.where(rating_matrix.iloc[0].isnull())[0]]
    #nan_indexes = np.where(rating_matrix.iloc[:,0].isna())[0]
    #missing_indeces.append(nan_indexes)
    ret = predict_ratings_user_based(0,missing_indices,column_vectors,rating_matrix)
    part_dict = dict()
    key = -1 
    part = sorted(parts[0], key = lambda x: x[1])
    for item in part:
        if key != item[0]:
            key = item[0]
            part_dict[key] = []
        part_dict[key].append(item)  
    sorted_ret = sorted(ret, key = lambda x: x[0])
    #with open("item_test_0.pickle","wb") as file:
        #pickle.dump(sorte_red,file)
    sum = 0
    counter = 0
    top_item = 0
    for item in part_dict[0]:
        for ret_i in sorted_ret:
            if item[0] == ret_i[0]:
                sum+=item[2] - ret_i[2]
                if ret_i[2] > top_item:
                    top_item = ret_i[2]
                print(f"prev:{item}; act:{ret_i}")

    for ret_i in sorted_ret:
        if ret_i[2] > top_item:
            counter+=1
    print(sum/len(part_dict[0]))
    print(top_item)
    print(counter)
"""
