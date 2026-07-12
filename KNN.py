import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

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
from sklearn.preprocessing import StandardScaler

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=3123)

# we need to sclae the features ,which means X - xtest,xtrain
#PREPROCESSING

scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
print(xtrain[:5])

"""[-0.18228655  0.88608836 -0.50126614  0.84775188  0.75291085  0.27208059
  -0.02411591  0.44838813  0.86202266 -0.01695481 -0.95039356 -0.34385802
  -0.69075612 -0.32931431 -0.15594131 -0.7671021  -0.57497937 -0.27640237
   0.1134293  -0.82378158 -0.53790885  0.72960402 -0.08392541  0.68946781
   0.57501276  0.38376595 -0.08095178  0.31949931  1.09078904  0.41939871
  -0.16566819]""" # The output after Sdandard Scaler , all the vlaues become same level approximately

xtest =  scaler.transform(xtest)
model = KNeighborsClassifier(n_neighbors=23) #k by find the root for 569, which is its no.of.rows
model.fit(xtrain,ytrain)

predict_prob = model.predict_proba(xtest)
print(predict_prob[:5]) # eg -> k=5  : 1 1 1 0 0 -> class 0 = 2/5 and class 1 = 3\5

predict = model.predict(xtest)
print(predict[:5])

# Model Evaluation 

print(f"Accuracy:{accuracy_score(ytest,predict)}")
print(f"Precision:{precision_score(ytest,predict)}")
print(f"Recall: {recall_score(ytest,predict)}")
print(f"F1-Score: {f1_score(ytest,predict)}")
print(f"Confusion Matrix: {confusion_matrix(ytest,predict)}")
print(f"Classification Report: {classification_report(ytest,predict)}")

"""#Actual vs predicted
import matplotlib.pyplot as plt
plt.scatter(ytest,predict)
plt.show()
#this is not useful for classification"""

