from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ds = pd.read_csv('insurance.csv')
#print(ds)

dummieSex = pd.get_dummies(ds['sex'])
dummieRegion = pd.get_dummies(ds['region'])
merged = pd.concat([ds,dummieSex,dummieRegion],axis='columns')
merged['smoker'] = merged['smoker'].replace({'no': 0, 'yes': 1})


#print(merged)
final = merged.drop(columns=['sex','region','charges'])
print(final.to_string())

#X_train, X_test, y_train, y_test = train_test_split(final,ds.charges, random_state=10)
model = linear_model.LinearRegression()
model.fit(final.values, ds.charges.values)
#print(model.predict(X_test))
#print(model.score(X_test, y_test))
print(model.predict([[18,33.770,1,0,0,1,0,0,1,0]]))
print(model.score(final.values, ds.charges.values))

"""
X_train, X_test, y_train, y_test = train_test_split(ds.area,ds.peri, random_state=10)

reg = linear_model.LinearRegression()

reg.fit([X_train[:]], y_train.values)
# First argument has to be 2D array
print(X_test)
# Now the machine will predict
#print(reg.predict([X_test]))

dummies = pd.get_dummies(df['Car Model'])

merged = pd.concat([df,dummies],axis='columns')

final = merged.drop(['Car Model','Mercedez Benz C class'],axis='columns')


X = final.drop('Sell Price($)',axis='columns')
Y = final['Sell Price($)']

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X.values, Y.values)

#print(model.score(X, Y))

print(model.predict([[45000,4,0,0]]))"""