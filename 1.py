import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


data=pd.read_csv("iris.csv")

h=["sepal.length", "sepal.width", "petal.length","petal.width","variety"]

X=data[h[0:4]].to_numpy()
y=data["variety"].to_numpy()


# plt.scatter(X[:50,0],X[:50,1],c="red")
# plt.scatter(X[51:100,0],X[51:100,1],c="blue")
# plt.scatter(X[101:150,0],X[101:150,1],c="green")
# plt.show()



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_hat=knn.predict(X_test)

from sklearn.metrics import accuracy_score

z=accuracy_score(y_test,y_hat)
print(z)