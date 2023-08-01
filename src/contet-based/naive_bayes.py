import numpy as np
import pandas as pd 
import os
import pickle
import sys
sys.path.append('../cross_validation/')
import cross_validation
import cProfile
import pstats
import multiprocessing 

#PATH_TO_MOVIES = '../../data_movilens/content-based movies.csv'
#PATH_TO_RATINGS = '../../data_movilens/content-based ratings.csv'
PATH_TO_RATINGS='../../data_movilens/ml-latest-small/ratings.csv'
PATH_TO_MOVIES = '../../data_movilens/ml-latest-small/movies.csv'
PATH_TO_CROSS = '../cross_validation_parts.pickle'

def get_indexes_of_not_empty_ratings_by_user(df):
    array_of_arrays = []
    for _,row in df.iterrows():
        non_nan_columns = [column for column, value in row.items() if pd.notnull(value)]
        array_of_arrays.append(non_nan_columns)
    return array_of_arrays

def generate_user_dataframe(rating_matrix, movies_df, user_id):
    #user_ratings_df = rating_matrix.iloc[user_id,rating_matrix.columns.isin(movies_id)]
    user_ratings_df = rating_matrix.iloc[user_id]
    user_ratings_df = user_ratings_df.to_frame('rating').reset_index()
    movie_id_indeces = user_ratings_df['movieId']
    filtered_movies_df = movies_df[movies_df['movieId'].isin(movie_id_indeces)]
    genres_df = filtered_movies_df['genres'].str.get_dummies('|')
    genres_id = pd.concat([filtered_movies_df['movieId'],genres_df], axis = 1)
    merged_df = pd.merge(genres_id, user_ratings_df, on='movieId', how='left')
    #return merged_df.drop(['userId','timestamp'],axis = 1)
    return merged_df

def pre_calculate_user_model(user_df,genres_column_names,values_of_genre,class_options):
    laplac_alpha = 0.1
    laplac_k = 2*laplac_alpha
    all_probabilities = []
    for genre in genres_column_names:
        positive_class_indexes = values_of_genre[genre][values_of_genre[genre] == 1].index
        negative_class_indexes = values_of_genre[genre][values_of_genre[genre] == 0].index
        positive_feature_count = positive_class_indexes.shape[0]
        negative_feature_count = negative_class_indexes.shape[0]
        wrap = [[],[]]
        positive_class_probabilities = ()
        negative_class_probabilities = () 
        for class_value in class_options:
            positive_class_count = (user_df.loc[positive_class_indexes,'rating'] == class_value).sum() 
            negative_class_count = (user_df.loc[negative_class_indexes,'rating'] == class_value).sum() 
            #print(f"pos_class:{(user_df.loc[positive_class_indexes,'rating'] == class_value).sum()},neg_class:{(user_df.loc[negative_class_indexes,'rating'] == class_value).sum()} class_value:{class_value},genre:{genre}")
            #positive_class_probabilities =  positive_class_probabilities + (np.log((positive_class_count + laplac_alpha)/(positive_feature_count + laplac_k)),)
            #negative_class_probabilities = negative_class_probabilities + (np.log((negative_class_count + laplac_alpha)/(negative_feature_count + laplac_k)),)
            positive_class_probabilities =  positive_class_probabilities + ((positive_class_count + laplac_alpha)/(positive_feature_count + laplac_k),)
            negative_class_probabilities = negative_class_probabilities + ((negative_class_count + laplac_alpha)/(negative_feature_count + laplac_k),)

        wrap[0] = negative_class_probabilities 
        wrap[1] = positive_class_probabilities 
        all_probabilities.append(wrap)
    return all_probabilities

def naive_bayes_prediciton(user_df,user_id,model):
    predictions = []
    genres_column_names = np.array(user_df.columns)
    genres_column_names = genres_column_names[2:len(genres_column_names)-1]
    nan_indexes = user_df[user_df['rating'].isna()].index 
    class_options = [1.0,2.0,3.0,4.0,5.0]
    values_of_genre = dict()
    laplac_alpha = 0.01
    laplac_k = 2 * laplac_alpha
    nan_vectors = []
    for nan_index in nan_indexes:
        nan_vectors.append(user_df.iloc[nan_index])
    for genre in genres_column_names:
        values_of_genre[genre] = user_df[genre].drop(index = nan_indexes)
    prior_class_probability= []
    for class_value in class_options:
            prior_class_probability.append( np.log(
                    (((user_df['rating'] == class_value).sum()+laplac_alpha)/((user_df['rating']).count()+laplac_k))))
    user_df = user_df.dropna()
    if model == 0:
        model = pre_calculate_user_model(user_df,genres_column_names,values_of_genre,class_options)
    for nan_vector in nan_vectors:
        nan_vector_shrinked = nan_vector[2:len(nan_vector)-1]
        sum = np.ones(5)
        for i,feature in enumerate(nan_vector_shrinked):
            binary_feature = int(feature)
            genre = model[i]
            result = genre[binary_feature]
            sum *= result
        base_probabilities = sum * np.exp(prior_class_probability)
        numerators = base_probabilities * class_options 
        predictions.append((numerators.sum()/base_probabilities.sum(),nan_vector[0]))
    return (predictions,user_id,model)
   
if __name__ == '__main__':
    ratings_df = pd.read_csv(PATH_TO_RATINGS,delimiter = ',')
    movies_df = pd.read_csv(PATH_TO_MOVIES,delimiter = ',')
    rating_matrix= pd.pivot_table(data=ratings_df,index="userId",columns="movieId", values="rating")
    rating_matrix = rating_matrix.reset_index(drop = True)
    rating_matrix.index.name = "userId"
    print(rating_matrix)
    #print(rating_matrix)
    profiler = cProfile.Profile()
    profiler.enable()
    parts = []
    if not os.path.exists(PATH_TO_CROSS):
        parts = cross_validation.create_parts_dataset(5,131,rating_matrix)
        with open(PATH_TO_CROSS,"wb") as file:
            pickle.dump(parts,file)
    else:
        with open(PATH_TO_CROSS,"rb") as file:
            parts = pickle.load(file)

    for iteration,part in enumerate(parts):
        rating_matrix_clone = rating_matrix.copy()
        for rating_tuple in part:
            row,column,rating=rating_tuple
            rating_matrix_clone.iloc[row,column] = np.nan 

        final_dataframe = rating_matrix_clone.copy()
        final_dataframe = pd.DataFrame(np.nan, index = rating_matrix_clone.index, columns = rating_matrix_clone.columns)
        iterable = []
        model = []
        if os.path.exists("../contet-based/naive_bayes_model"+str(iteration)+".pickle"):
            with open("../contet-based/naive_bayes_model"+str(iteration)+".pickle",'rb') as file:
                model = pickle.load(file)
        
        for i,_ in rating_matrix_clone.iterrows():
            user_data = generate_user_dataframe(rating_matrix_clone,movies_df,i)
            model_value = 0
            if len(model) > 0:
                model_value = model[i]
            iterable.append((user_data,i,model_value))

        pool = multiprocessing.Pool(processes=8)
        predictions = pool.starmap(naive_bayes_prediciton, iterable)
        pool.close()
        pool.join()

        profiler.disable()
        profiler.dump_stats('profile_stats')
        # Create a pstats.Stats object
        model = []
        for predicted_list,user_id,user_model in predictions:
            model.append(user_model)
            for predicted_value,movie_id in predicted_list:
                final_dataframe.loc[user_id,movie_id] = predicted_value
        with open("naive_bayes_prediciton"+str(iteration)+".pickle","wb") as file:
            pickle.dump(final_dataframe,file)

        with open("naive_bayes_model"+str(iteration)+".pickle","wb") as file:
            pickle.dump(model,file)
    
    stats = pstats.Stats('profile_stats')
    stats.print_stats()


