import random as rnd
import numpy as np
import pandas as pd


arr = [[1,2,3],[1,1,1]]
s = pd.Series(arr[0])
s2 = pd.Series(arr[1])
print(np.sum(-2*(s-s2)))


