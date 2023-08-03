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
PATH_TO_RESULTS_SVD = '../collaborative-filtering/results/model_50/'
PATH_TO_RESULTS_MEMORY_BASED= '../collaborative-filtering/results/neighbour/'
PATH_TO_RESULTS_CONTENT_BASED = '../contet-based/results'
PATH_TO_RESULTS_HYBRID = '../hybrid/results'
PATH_TO_PARTS = '../cross_validation_parts.pickle'
PATH_TO_RESULTS= './results.csv'

    
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

def top_k_evalutaion(five_star_rating_cells,predicted_row,original_row):
    rnd.seed(1000)
    not_rated_indeces = original_row.index[original_row.isna()].values 
    hit = 0
    for five_star_rating in five_star_rating_cells:
        _,column,_= five_star_rating
        nri_copy = not_rated_indeces.copy()
        chosen_indexes = []
        for _ in range(500):
            rnd_int = rnd.randint(0,len(nri_copy)-1) 
            if rnd_int == column:
                nri_copy = np.delete(nri_copy,rnd_int)
                rnd_int = rnd.randint(0,len(nri_copy)-1)
            predicted_index = nri_copy[rnd_int]
            if np.isnan(predicted_row.iloc[predicted_index]):
                nri_copy = np.delete(nri_copy,rnd_int)
                rnd_int = rnd.randint(0,len(nri_copy)-1)
                predicted_index = nri_copy[rnd_int]
            chosen_indexes.append(predicted_index)
            nri_copy = np.delete(nri_copy,rnd_int)

        chosen_values = predicted_row.iloc[chosen_indexes] 
        chosen_values = pd.concat([chosen_values,pd.Series([predicted_row.iloc[column]], index = [-1])])
        sorted_list = chosen_values.sort_values(ascending = False)
        top_20 = sorted_list[:20]
        if -1 in top_20.index.values: 
            hit += 1
    return hit,len(five_star_rating_cells)
        

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
    path_to_results = PATH_TO_RESULTS_SVD

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
            rating_matrix_clone= rating_matrix.copy()
            df = recommendation
            numbers_array = [num for num in range(0, df.shape[1])]
            df.columns = numbers_array
            rating_matrix_clone.columns = numbers_array
            df = df.reset_index(drop = True)
            df.index.name = "userId"
            keys = bundle_dict.keys()
            counter = hits = test_cases = novelty_ret = succesful_recommendation = 0
            for j, row in df.iterrows(): 
                if j in keys:
                    hits_ret, test_cases_ret= top_k_evalutaion(bundle_dict[j],df.iloc[j],rating_matrix_clone.iloc[j]) 
                    hits += hits_ret
                    test_cases += test_cases_ret
                    counter += 1
                    sorted_row = df.iloc[j].sort_values(ascending = False)
                    top_20 = sorted_row.iloc[:20]
                    if len(top_20) == 20:
                        succesful_recommendation += 1
                    else:
                        print('yo')
                    novelty_ret += novelty(df,top_20)

            recall = hits/test_cases
            precision = recall / 20

            novelty_result = novelty_ret / counter
            f1_result = f1_score(precision, recall)
            error_values = get_error_values(part_dict,df)
            rmse = RMSE(error_values,len(parts[part_number]))
            mae = MAE(error_values,len(parts[part_number]))
            coverage = user_coverage(succesful_recommendation, df.shape[0])
            with open(PATH_TO_RESULTS, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow((filename,precision,recall,f1_result,novelty_result,coverage,rmse,mae))    

