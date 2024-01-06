import numpy as np
import pandas as pd
import pickle 

number_of_iter = '1'
name_of_file = 'test_i_model_svd_20_'+number_of_iter+'.pickle'
name_of_results = './results/model/i_model_svd_'+number_of_iter+'.pickle'
dataframe = None
dataframe2 = None
for i in range(4):
    with open(name_of_file,'rb') as file: 
        dataframe = pickle.load(file)
    with open(name_of_results,'rb') as file:
        dataframe2 = pickle.load(file)
    print(dataframe.equals(dataframe2))
    


