import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split

from sklearn import tree
clf = tree.DecisionTreeClassifier()

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder
data=pd.read_csv("drug200.csv")


label_encoder = LabelEncoder()
data["data_Sex"]=label_encoder.fit_transform(data["Sex"])

label_encoder = LabelEncoder()
data["data_BP"]=label_encoder.fit_transform(data["BP"])

label_encoder = LabelEncoder()
data["data_Cholesterol"]=label_encoder.fit_transform(data["Cholesterol"])

X=data[["Age","data_Sex","data_BP","data_Cholesterol","Na_to_K"]]
y=data["Drug"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

clf.fit(X_train,y_train)

y_hat = clf.predict(X_test)

ac=accuracy_score(y_test,y_hat)
print(ac)