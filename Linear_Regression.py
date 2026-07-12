import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df  = pd.read_csv("data//housing.csv")


print (df.columns) #To extract the columns from the dateset
print(df.head()) # Returns the first 5 rows from the dataset
print(df.describe())
print(df.info())
df=df.drop("ocean_proximity",axis=1)
print("After dropping object coloumn")
print(df)

print(df.isnull())
print(df.isnull().sum())

# To drop the rows which has missing values (total_bedrooms-> missing vlaues->207)
# df = df.dropna(), but we don't want to drop the rows simply
# Better we can use the mean values of the Specific coloums named "total_bedrooms" to fill the Null Values

df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"]).mean()
print("Null values Filled with their mean values...")
print(df.isnull().sum())

# To check duplicates 
print("Duplicated Values.......................")
print(df.duplicated().sum())
print(df.drop_duplicates())
print("Duplicated Values removed")
print("Shape...................................")
print(df.shape)
print(df["total_bedrooms"])

x = df.drop("median_house_value",axis=1)
y = df["median_house_value"]

print(x.head())
print(y.head())

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.25,random_state=3123)

model = LinearRegression()
model.fit(xtrain,ytrain)
predictions = model.predict(xtest)
print("predictons...................................")
print(predictions)

#Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(ytest,predictions)
mse = mean_squared_error(ytest,predictions)
r2  = r2_score(ytest,predictions)

print(f"MSE : {mse}\nMAE : {mae}\nR2 : {r2}")

#Actual vs predicted
plt.scatter(ytest,predictions)
plt.show()