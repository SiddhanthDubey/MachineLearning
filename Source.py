import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('kc_house_data.csv')
df = df.drop(columns=['id', 'date'])
#print(df)

columns = np.array(df.columns)
columns = np.delete(columns,0)

reg = linear_model.LinearRegression()
reg.fit(df[columns].values, df.price.values)
j = [3,1.00,1180,5650,1.0,0,0,3,7,1180,0,1955,0,98178,47.5112,-122.257,1340,5650]
print(columns)
print(reg.predict([j]))
r_squared = reg.score(np.array(df[columns].values), np.array(df.price.values))
print(r_squared)
