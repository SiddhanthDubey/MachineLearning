import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
digits = load_digits()
print(dir(digits))

print(digits.data[0])

plt.gray()
plt.matshow(digits.images[0])
plt.show()

print(digits.target[0:5])

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.1)

print(len(x_train))

model = LogisticRegression()
model.fit(x_train,y_train)

print(model.score(x_test, y_test))
predicted = model.predict(x_test)
print(predicted)
