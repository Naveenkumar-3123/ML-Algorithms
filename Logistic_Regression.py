import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score,recall_score,f1_score
import matplotlib

df = pd.read_csv("data//titanic.csv")
print(df.columns)
print(df.head())

# we dont need the name of the passenger, since name doesn't give sence for survival
df=df.drop("Name",axis=1)

print("Columns.......................\n")
print(df.columns)
print("Info............................")
print(df.info())
print("Shape.............................")
print(df.shape)
print("Describe..................................")
print(df.describe())
print(df.dtypes)
print("Total null.............................")
print(df.isnull().sum())

# Now the age has 177 null values , we can fill it with its mean

df["Age"] = df["Age"].fillna(df["Age"]).mean()


# we dont need the cabin of the passenger, since name doesn't give sence for survival

df = df.drop("Cabin",axis=1)
print("Cabin colums droppped...................")
df = df.drop("Ticket",axis=1)
print("Ticket colums droppped...................")

#Embarked has missng values of two , so that we fill it using mode , scince the dtype is string/object
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

#We need to map the Gender and Embarked

df["Sex"] = df["Sex"].map({"female":0,"male":1})
df["Embarked"] = df["Embarked"].map({"C":0,"S":1,"Q":2})

x = df.drop("Survived",axis=1)
y = df["Survived"]

from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression

print(x.dtypes)
print(x.isnull().sum())
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2,random_state = 3123)

model = LogisticRegression(max_iter=1000)
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


"""#Visuvalize actual vs predicted

import matplotlib.pyplot as plt
plt.scatter(ytest,predict)
plt.show() 
#this is not useful for classification
"""


