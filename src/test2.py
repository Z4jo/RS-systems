import pandas as pd
data = [
    [25,1,2,3,6],
    [30,1,2,3,4],
    [22,3,5,6,6],
    [42,3,36,6,6]
] 

df = pd.DataFrame(data)

print(df)
test = df[{1,2}]
print(test)

