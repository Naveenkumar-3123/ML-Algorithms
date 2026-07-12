import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
 
df = pd.read_csv("data//WineQT.csv")

print(df.columns)
print(df.shape)
print(df.info())
print(df.isnull().sum())
# No null values
print(df.duplicated().sum())

print(df.describe())
#we dont need id
df= df.drop("Id",axis=1)
print("Id dropped...........................")

x = df.drop("fixed acidity",axis=1)
y = df["fixed acidity"]
# Here our predictions are continuous,so use regressor
print(df.dtypes)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
#We are going to use the regressor here, so we need to use mse,mae and r2
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=3123)
rank=[]
for i in [3,4,5,6]:
    """model = DecisionTreeClassifier(max_depth=i)
    model.fit(xtrain,ytrain)
    predict = model.predict(xtest)
    accuracy = accuracy_score(ytest,predict)
    precision = precision_score(ytest,predict)

    recall = recall_score(ytest,predict)
    f1 = f1_score(ytest,predict)
    rank.append([i,accuracy,precision,recall,f1])"""
    model = DecisionTreeRegressor(max_depth=i,random_state=3123)
    model.fit(xtrain,ytrain)
    predict = model.predict(xtest)
    mse = mean_squared_error(ytest,predict)
    mae = mean_absolute_error(ytest,predict)
    r2 = r2_score(ytest,predict)
    rank.append([i,mse,mae,r2])

result = pd.DataFrame(rank,columns=["Max_Depth","MSE","MAE","R2"])

print(result)

#Actual Vs precited

import matplotlib.pyplot as plt
plt.scatter(ytest,predict)
plt.title("Acutal VS Predicted")
plt.show()
