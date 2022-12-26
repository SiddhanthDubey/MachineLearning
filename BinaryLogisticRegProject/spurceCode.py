import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

df = pd.read_csv('HR_comma_sep.csv')
print(df.columns)
df = df.drop(['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Department'],axis='columns')
print(df.columns)
dummies = pd.get_dummies(df['salary'])
df = df.drop(['salary'],axis='columns')
df = pd.concat([df,dummies],axis='columns')
print(df.columns)


x_train, x_test, y_train, y_test = train_test_split(df[['Work_accident', 'promotion_last_5years', 'high', 'low', 'medium']],df.left,test_size=0.1)

print(x_test)

model = LogisticRegression()
model.fit(x_train, y_train)

print(model.predict(x_test))

print(model.score(x_test, y_test))