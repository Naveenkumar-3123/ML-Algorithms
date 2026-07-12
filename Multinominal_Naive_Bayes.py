import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("data\\sms+spam+collection\\SMSSpamCollection",sep="\t",header=None)
df.columns = ["label","messages"]
print(df.info())
print(df.head())
print(df.describe())
print("Duplicates...............")
print(df.duplicated().sum())
"""
Duplicates...............
403
"""
df = df.drop_duplicates()
print(df.duplicated().sum())
# Now th duplicates = 0

x = df["messages"]
y = df["label"]


vectorize = CountVectorizer()
vectorize.fit(x)
x = vectorize.transform(x)
print(x)
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=3123)
# we vectorized the X before spliting to xtrain, so that x train aslo looks the SAME and npt need to vectorize;
print(xtrain)

model = MultinomialNB()
model.fit(xtrain,ytrain)
predict =model.predict(xtest)
print(predict[:5])

print("Accuracy: \n",accuracy_score(ytest,predict))
print("Confusion Matrix: \n",confusion_matrix(ytest,predict))
print("classification Report :\n",classification_report(ytest,predict))
