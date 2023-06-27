import random as rnd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
PATH_TO_RATINGS='../data_movilens/ml-latest-small/ratings.csv'
PATH_TO_MOVIES = '../data_movilens/ml-latest-small/movies.csv'

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

ratings_df = pd.read_csv(PATH_TO_RATINGS,delimiter = ',')
movies_df = pd.read_csv(PATH_TO_MOVIES,delimiter = ',')
rating_matrix= pd.pivot_table(data=ratings_df,index="userId",columns="movieId", values="rating")
rating_matrix = rating_matrix.reset_index(drop = True)
rating_matrix.index.name = "userId"
user_dataframe = generate_user_dataframe(rating_matrix,movies_df,0)
nan_indexes = user_dataframe[user_dataframe['rating'].isna()].index 
nan_vectors = []
for nan_index in nan_indexes:
    full_vector = user_dataframe.iloc[nan_index]
    nan_vectors.append(full_vector[2:21])
nan_df = pd.DataFrame(nan_vectors)
print(nan_df)

user_dataframe = generate_user_dataframe(rating_matrix,movies_df,0)
user_dataframe = user_dataframe.dropna()
X = user_dataframe.iloc[:,2:21]
y = user_dataframe.iloc[:,21]
#X_train, X_test, y_train, y_test =train_test_split(X,y)
#print(X_test)
#print(X_train)
#sc_X = StandardScaler() 
#X_train = sc_X.fit_transform(X)
#X_test = sc_X.fit_transform(X)
classifer = BernoulliNB()
# training the model
#print(X)
classifer.fit(X, y)
# testing the model

y_pred = classifer.predict(nan_df)
std = y_pred.std()
print(len(y_pred))
print(y_pred)
print(std)

#print(accuracy_score(y_pred, y_test))

# create a Gaussian Classifier
classifer1 = GaussianNB()
# training the model
# testing the model
# printing the accuracy of the model

