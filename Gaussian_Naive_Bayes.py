import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("data//Breast_cancer_wisconsin.csv")
print("coloumns....................")
print(df.columns)
print(df.dtypes)
print(df.head())

# since diagonisis have the object dtype , so we need to find the unique values to map to interger 

print("Unique values..................................")
print(df["diagnosis"].unique())

# here the result is M and B ,so we need to map it has 0 and 1
# M-> malignant(Can be spread-dangerous) and B -> Benign(Non-thretened life)

df["diagnosis"] = df["diagnosis"].map({"M":1,"B":0})
print("Dtypes after mapping")
print(df.dtypes)
print(df.shape)

# and aslo i found that the last colums has no name with nan values , so that we need to drop it..

df = df.drop("Unnamed: 32",axis=1)
print("Anfter dopping last coloumn..................")
print(df.head())
print("Null values .....................................")
print(df.isnull().sum())
# that's good , we dont have null values anywhere
print("Duplicates.......................................")
print(df.duplicated().sum())
# we have 0 duplicated values..

x = df.drop("diagnosis",axis=1)
y = df["diagnosis"]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=3123)
print(x.dtypes)
print(x.isnull().sum())
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2,random_state = 3123)

model = GaussianNB()
model.fit(xtrain,ytrain)
#Let use the proba to understand the class division .
predict_proba = model.predict_proba(xtest)
print("Probabilities...............")
print(predict_proba)

predict = model.predict(xtest)
print(predict)

print(f"Accuracy:{accuracy_score(ytest,predict)}")
print(f"Precision:{precision_score(ytest,predict)}")
print(f"Recall: {recall_score(ytest,predict)}")
print(f"F1-Score: {f1_score(ytest,predict)}")
print(f"Confusion Matrix: {confusion_matrix(ytest,predict)}")
print(f"Classification Report: {classification_report(ytest,predict)}")