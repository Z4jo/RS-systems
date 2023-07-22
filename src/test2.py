import pandas as pd
import numpy as np
import math
import sys
import pickle
import os 

PATH_TO_RESULTS = './contet-based/results/'

for index, filename in enumerate(os.listdir(PATH_TO_RESULTS)):
            filepath = os.path.join(PATH_TO_RESULTS, filename)
            print(filename)
            recommendation = 0
            with open(filepath, 'rb') as file:
                recommendation = pickle.load(file)
            print(recommendation)
            

