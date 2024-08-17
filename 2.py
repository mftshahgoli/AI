import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import jaccard_score,confusion_matrix


data=pd.read_csv("titanic.csv")
label_encoder = LabelEncoder()
data["newData"]=label_encoder.fit_transform(data["Sex"])


X=data[["Pclass","newData"]]
y=data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)


knn=KNeighborsClassifier(n_neighbors=18)
knn.fit(X_train,y_train)
y_hat=knn.predict(X_test)

z=confusion_matrix(y_test,y_hat)
print(z)







# L=[]
# for k in range(1,50):
#     knn=KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train,y_train)
#     y_hat=knn.predict(X_test)
#     z=jaccard_score(y_test,y_hat)
#     L.append(z)

# print(L)
# plt.plot(L)
# plt.show()