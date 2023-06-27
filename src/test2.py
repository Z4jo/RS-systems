import numpy as np

d = dict()
d[0] = 2
d[1] = 2
d[1] = 2
t = (1,1,1)
d = (2,2,2)
a = [0,0,0]

result = a + np.array(t)
print(result)
result +=  np.array(t)
result = result * d
print(result)
t = t + d
print(t)
