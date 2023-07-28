from datetime import time
import pandas as pd
import numpy as np
import math
import os
import pickle
import csv
import matplotlib.pyplot as plt
import seaborn as sns


PATH_TO_DATA = '../../data_movilens/ml-latest-small/ratings.csv'
#NOTE: changes based on results we want to analyze
PATH_TO_RESULTS_SVD = '../collaborative-filtering/results/model_based_svd/'
PATH_TO_RESULTS_MEMORY_BASED= '../collaborative-filtering/results/memory_based/'
PATH_TO_RESULTS_CONTENT_BASED = '../contet-based/results'
PATH_TO_RESULTS_HYBRID = '../hybrid/results'
PATH_TO_PARTS = '../collaborative-filtering/cross_validation_parts.pickle'
PATH_TO_RESULTS= './results.csv'

    
def precision(timestamp_matrix, top_k_list, user_index):
    indices = top_k_list.index
    tmp = 0
    for index, _ in enumerate(top_k_list):
        if  not pd.isna(timestamp_matrix.iloc[user_index,indices[index]]):
            tmp += 1
    return tmp / len(top_k_list)

def recall(user_part, top_k_list, user_index ):
    indices = top_k_list.index
    tmp = 0
    for index, _ in enumerate(top_k_list):
        if  not pd.isna(timestamp_matrix.iloc[user_index,indices[index]]):
            tmp += 1
    return tmp / len(user_part)

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
                predicted_r = predicted_row.iloc[actual_r[1]]
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


if __name__ == '__main__':
    dataframe=pd.read_csv(PATH_TO_DATA,delimiter=',')
    rating_matrix= pd.pivot_table(data=dataframe,index="userId",columns="movieId", values="rating")
    timestamp_matrix= pd.pivot_table(data=dataframe,index="userId",columns="movieId", values="timestamp")
    numbers_array = [num for num in range(0, timestamp_matrix.shape[1])]
    timestamp_matrix.columns = numbers_array
    timestamp_matrix= timestamp_matrix.reset_index(drop = True)
    timestamp_matrix.index.name = "userId"


    print(timestamp_matrix)
    splited_path = PATH_TO_RESULTS_SVD.split('/') 
    splited_type_algo = splited_path[-2].split('_')
    parts = []
    
    if os.path.exists(PATH_TO_PARTS):
        with open(PATH_TO_PARTS, 'rb') as file:
            parts = pickle.load(file)
    else:
         raise ValueError("Invalid path to cross_validation_parts: " + PATH_TO_PARTS)
    if os.path.exists(PATH_TO_RESULTS_SVD):
        recommendation = 0
        num_rows = 610
        num_cols = 9742
        for index, filename in enumerate(os.listdir(PATH_TO_RESULTS_SVD)):
            filename_split = filename.split('.')
            part_number = int(filename_split[0][-1])
            filepath = os.path.join(PATH_TO_RESULTS_SVD,filename)
            with open(filepath, 'rb') as file:
                recommendation = pickle.load(file)
            part_dict = create_dict_from_part(parts[part_number])  
            if splited_type_algo[0] == "memory":
                print(filename)
                df = rating_matrix.copy()
                df = pd.DataFrame(index=range(num_rows), columns=range(num_cols))
                df = df.fillna(np.nan)
                for array in recommendation:
                    for row, column, rating in array:
                        if rating < 0:
                            rating = 0
                        elif rating > 5:
                            rating = 5
                        df.iloc[row, column] = rating
                denominator = f1_ret = precision_ret = recall_ret = novelty_ret = succesful_recommendation = 0
                keys = part_dict.keys()
                for j , row in df.iterrows():
                    if index in keys:
                        sorted_row = row.sort_values(ascending = False, na_position = "last")
                        top_k_ratings = sorted_row[:20]
                        if len(top_k_ratings) >= 20:
                            succesful_recommendation += 1
                        recall_ret += recall(part_dict[j], top_k_ratings, j)
                        precision_ret += precision(timestamp_matrix,top_k_ratings,j)
                        novelty_ret += novelty(df, top_k_ratings )
                        f1_ret += f1_score(precision_ret, recall_ret)
                        denominator += 1
                precision_result =  precision_ret / denominator
                recall_result = recall_ret / denominator
                novelty_result = novelty_ret / denominator
                f1_result = f1_score(precision_ret / denominator, recall_ret / denominator)
                error_values = get_error_values(part_dict,df)
                rmse = RMSE(error_values,len(parts[part_number]))
                mae = MAE(error_values,len(parts[part_number]))
                coverage = user_coverage(succesful_recommendation, df.shape[0])
                with open(PATH_TO_RESULTS, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow((filename,precision_result,recall_result,f1_result,novelty_result,coverage,rmse,mae))    


            else:
                print(filename)
                df = recommendation
                print(df)
                numbers_array = [num for num in range(0, df.shape[1])]
                df.columns = numbers_array
                df = df.reset_index(drop = True)
                df.index.name = "userId"
                keys = part_dict.keys()
                denominator = precision_ret = recall_ret = novelty_ret = succesful_recommendation = 0
                for j, row in df.iterrows(): 
                    if index in keys:
                        sorted_row = row.sort_values(ascending = False, na_position = "last")
                        top_k_ratings = sorted_row[:20]
                        if len(top_k_ratings) >= 20:
                            succesful_recommendation += 1
                        recall_ret += recall(part_dict[j], top_k_ratings, j)
                        precision_ret += precision(timestamp_matrix,top_k_ratings,j)
                        novelty_ret += novelty(df, top_k_ratings )
                        denominator += 1
                precision_result =  precision_ret / denominator
                recall_result = recall_ret / denominator
                novelty_result = novelty_ret / denominator
                f1_result = f1_score(precision_ret / denominator, recall_ret / denominator)
                error_values = get_error_values(part_dict,df)
                rmse = RMSE(error_values,len(parts[part_number]))
                mae = MAE(error_values,len(parts[part_number]))
                coverage = user_coverage(succesful_recommendation, df.shape[0])
                with open(PATH_TO_RESULTS, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow((filename,precision_result,recall_result,f1_result,novelty_result,coverage,rmse,mae))    


    else:
        raise ValueError("Invalid path to results: " + PATH_TO_RESULTS)
"""
    sorted_row = df.iloc[0].sort_values(ascending = False, na_position = "last")
                indexes = sorted_row.index
                print(df.iloc[0,0])
                print(part_dict[0])
                top_k_ratings = sorted_row[:100]
                ret_precision= precision(timestamp_matrix,top_k_ratings,0)

                with open(PATH_TO_RESULTS, 'a') as file:
                    file.write(filename + "\n"
                            +"precision:"+ str(precision_result) + "\n"
                               +"recall:" + str(recall_result) + "\n"
                               +"novelty:" + str(novelty_result) + "\n"
                               +"f1:" + str(f1_result) + "\n"
                               +"rmse:" + str(rmse) + "\n"
                               +"mae:" + str(mae) + "\n"
                               +"coverage:" + str(coverage) + "\n"
                               )
          

"""
