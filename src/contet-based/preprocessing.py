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

def naive_bayes_prediciton(user_df,user_id):
    print(user_id)
    predictions = []
    genres_column_names = np.array(user_df.columns)
    genres_column_names = genres_column_names[1:len(genres_column_names)-1]
    nan_indexes = user_df[user_df['rating'].isna()].index 
    class_options = [1.0,2.0,3.0,4.0,5.0]
    values_of_genre = dict()
    #TODO: find good alpha for movielens dataset
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
    #NOTE: structure of model = [[[[],[],[],[],[]],[]],[[],[],[],[],[]]]
    model = pre_calculate_user_model(user_df,genres_column_names,values_of_genre,class_options)
  #  print(model)
    for nan_vector in nan_vectors:
  #      print(nan_vector)
        nan_vector_shrinked = nan_vector[2:len(nan_vector)-1]
        sum = np.ones(5)
        #sum = np.zeros(5)
        for i,feature in enumerate(nan_vector_shrinked):
            binary_feature = int(feature)
            genre = model[i]
            result = genre[binary_feature]
            sum *= result
        #base_log_probabilities = sum + prior_class_probability
        base_probabilities = sum * prior_class_probability
        #base_probabilities = np.exp(base_log_probabilities) 
        numerators = base_probabilities * class_options 
   #     print(f"sum:{sum},\nbase_probabilities:{base_probabilities},\nnumerators:{numerators}")
        predictions.append((numerators.sum()/base_probabilities.sum(),nan_vector[0]))
    return (predictions,user_id)
   
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
    if not os.path.exists("./cross_validation_parts.pickle"):
        parts = cross_validation.create_parts_dataset(5,131,rating_matrix)
        with open("cross_validation_parts.pickle","wb") as file:
            pickle.dump(parts,file)
    else:
        with open("cross_validation_parts.pickle","rb") as file:
            parts = pickle.load(file)

    for iteration,part in enumerate(parts):
        rating_matrix_clone = rating_matrix.copy()
        for rating_tuple in part:
            row,column,rating=rating_tuple
            rating_matrix_clone.iloc[row,column] = rating

        all_not_nan_indeces= get_indexes_of_not_empty_ratings_by_user(rating_matrix_clone)
        final_dataframe = rating_matrix_clone.copy()
        final_dataframe = pd.DataFrame(np.nan, index = rating_matrix_clone.index, columns = rating_matrix_clone.columns)
        iterable = []
        
        for user_index,_ in enumerate(all_not_nan_indeces):
            user_data = generate_user_dataframe(rating_matrix_clone,movies_df,user_index)
            iterable.append((user_data,user_index))

        #user_dataframe = change_rating_to_class(user_index,user_dataframe,rating_matrix) 
        pool = multiprocessing.Pool(processes=8)
        predictions = pool.starmap(naive_bayes_prediciton, iterable)
        pool.close()
        pool.join()

        profiler.disable()
        profiler.dump_stats('profile_stats')
        # Create a pstats.Stats object
        for predicted_list in predictions:
            for predicted_value in predicted_list[0]:
                final_dataframe.loc[predicted_list[1],predicted_value[1]] = predicted_value[0]
        #print(rating_matrix)
        with open("naive_bayes_prediciton"+str(iteration)+".pickle","wb") as file:
            pickle.dump(final_dataframe,file)
        stds = final_dataframe.std(axis = 0)
        print(stds.mean())
    
    stats = pstats.Stats('profile_stats')
    stats.print_stats()

    """
    iterable = []
    for user_index, missing_entries in enumerate(missing_indices):
        iterable.append((user_index, missing_entries ,item_rating_indexes,rating_matrix))
    pool = multiprocessing.Pool(processes=8)
    results = pool.starmap(predict_ratings_user_based,iterable)
    pool.close()
    pool.join()
    """
    """
    parts = []
    if not os.path.exists("./cross_validation_parts.pickle"):
        parts = cross_validation.create_parts_dataset(5,131,rating_matrix)
        with open("cross_validation_parts.pickle","wb") as file:
            pickle.dump(parts,file)
    else:
        with open("cross_validation_parts.pickle","rb") as file:
            parts = pickle.load(file)

    print(ratings_df)
    print(movies_df)
    for iteration,part in enumerate(parts):
        rating_matrix_clone= rating_matrix.copy()
        for rating_tuple in part:
            row,column,rating=rating_tuple
            rating_matrix_clone.iloc[row,column] = rating
            """

