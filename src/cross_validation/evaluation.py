import pandas as pd
import numpy as np
import math
import os
import pickle
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import random as rnd


PATH_TO_DATA = '../../data_movilens/ml-latest-small/ratings.csv'
#NOTE: changes based on results we want to analyze
PATH_TO_RESULTS_SVD = '../collaborative-filtering/results/model/'
PATH_TO_RESULTS_MEMORY_BASED= '../collaborative-filtering/results/neighbour/'
PATH_TO_RESULTS_CONTENT_BASED = '../contet-based/results'
PATH_TO_RESULTS_HYBRID = '../hybrid/results'
PATH_TO_PARTS = '../cross_validation_parts.pickle'
#WARN: change this
PATH_TO_RESULTS= './evaluation_2.0.csv'
#PATH_TO_RESULTS= './results.csv'

    
def f1_score(precision, recall):
    nominator = 2 * precision * recall
    denominator = precision + recall
    if denominator == 0 or nominator == 0:
        return 0
    return nominator / denominator

def RMSE(error_values, part_lenght):
    nominator = sum(x**2 for x in error_values)
    return math.sqrt(nominator / part_lenght)

def get_error_values(part_dict,predicted_rating_df):
    error_values = []
    for user_key in part_dict:
        predicted_row = predicted_rating_df.iloc[user_key]
        actual_row = part_dict[user_key]
        for actual_r in actual_row:
            predicted_r = predicted_row.loc[actual_r[1]]
            if not pd.isna(predicted_r):
                error_values.append(predicted_r - actual_r[2])  
    return error_values            

def MAE(error_values, part_lenght):
    nominator = sum(abs(x) for x in error_values)
    return nominator / part_lenght

def novelty(df, top_k_ratings):
    #pop == number of users that consumes it == numbers of users that rated the item     
    indeces = top_k_ratings.index
    df_count = df.count(axis = 0)
    result = 0
    for index,_ in enumerate(top_k_ratings):
        pop = df_count[indeces[index]]
        result += math.log2(pop)/20
    return result 

def user_coverage(succesful_recommendation, all_users_count):
    return abs(succesful_recommendation) / all_users_count

def create_dict_from_part(part):
    sorted_part = sorted(part, key = lambda x: x[0])
    users_ratings_dict = dict()
    key = -1
    for tuple in sorted_part:
        if key != tuple[0]:
            key = tuple[0]
            users_ratings_dict[key] = []
        users_ratings_dict[key].append(tuple)
    return users_ratings_dict

def hit_rate_calculation( relevant_indexes, top_ratings):
    hit = 0
    for _,index,_ in relevant_indexes:
        if index in top_ratings.index:
            hit+=1 
    return hit

def combine_test_indexes(all_test_items):
    item_indexes = set()
    for user in all_test_items: 
        for item in all_test_items[user]:
            item_indexes.add(item[1])
    return item_indexes

def generate_recommendation_test_set(recommendation,indexes):
    numbers_array = [num for num in range(0, recommendation.shape[1])]
    recommendation.columns = numbers_array
    rating_matrix_clone.columns = numbers_array
    recommendation = recommendation.reset_index(drop = True)
    recommendation.index.name = "userId"
    return recommendation[indexes]

if __name__ == '__main__':
    dataframe=pd.read_csv(PATH_TO_DATA,delimiter=',')
    rating_matrix= pd.pivot_table(data=dataframe,index="userId",columns="movieId", values="rating")
    timestamp_matrix= pd.pivot_table(data=dataframe,index="userId",columns="movieId", values="timestamp")

    numbers_array = [num for num in range(0, timestamp_matrix.shape[1])]
    timestamp_matrix.columns = numbers_array
    rating_matrix.columns =numbers_array
    timestamp_matrix= timestamp_matrix.reset_index(drop = True)
    timestamp_matrix.index.name = "userId"
    #WARN: CHANGE FOR DIFFERENT ALGO EVALUATION
    path_to_results = PATH_TO_RESULTS_MEMORY_BASED

    splited_path = path_to_results.split('/') 
    splited_type_algo = splited_path[-2].split('_')
    parts = []
    
    if os.path.exists(PATH_TO_PARTS):
        with open(PATH_TO_PARTS, 'rb') as file:
            parts = pickle.load(file)
    else:
         raise ValueError("Invalid path to cross_validation_parts: " + PATH_TO_PARTS)
    if os.path.exists(path_to_results):
        recommendation = 0
        for index, filename in enumerate(os.listdir(path_to_results)):
            filename_split = filename.split('.')
            part_number = int(filename_split[0][-1])
            bundle_dict = 0
            with open ("bunde_"+str(part_number)+".pickle","rb") as file:
                bundle_dict = pickle.load(file)
            filepath = os.path.join(path_to_results,filename)
            with open(filepath, 'rb') as file:
                recommendation = pickle.load(file)
            part_dict = create_dict_from_part(parts[part_number])  

            print(filename)
            rating_matrix_clone = rating_matrix.copy()
            keys = bundle_dict.keys()
            precision = recall = counter = hits = test_cases = novelty_ret = succesful_recommendation = 0
            test_movies_indexes = sorted(list(combine_test_indexes(part_dict)))
            recommendation = generate_recommendation_test_set(recommendation,test_movies_indexes)
            for user, row in recommendation.iterrows(): 
                if user in keys:
                    sorted_row = recommendation.iloc[user].sort_values(ascending = False)[:20]
                    hits = hit_rate_calculation(bundle_dict[user], sorted_row) 
                    if len(sorted_row) == 20:
                        succesful_recommendation += 1
                    novelty_ret += novelty(recommendation,sorted_row)
                    # true positives / true positives + false negatives
                    recall += hits/len(bundle_dict[user])
                    # true positives / true positives + false positives
                    precision += hits / 20
                    counter += 1

            precision = precision/counter 
            recall = recall/counter 
            novelty_result = novelty_ret / counter
            f1_result = f1_score(precision, recall)
            error_values = get_error_values(part_dict,recommendation)
            rmse = RMSE(error_values,len(parts[part_number]))
            mae = MAE(error_values,len(parts[part_number]))
            coverage = user_coverage(succesful_recommendation, recommendation.shape[0])
            with open(PATH_TO_RESULTS, 'a', newline = '') as file:
                writer = csv.writer(file)
                writer.writerow((filename,precision,recall,f1_result,novelty_result,coverage,rmse,mae))    

