import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score , confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
 
df = pd.read_csv("data//mushroom//agaricus-lepiota.data",header=None) 
print(df.columns) # With header = none ,the columns name is 0,1,2,3,4,5,6..22 or wthout header = the first row is columns name
df.columns=[
"class",
"cap_shape",
"cap_surface",
"cap_color",
"bruises",
"odor",
"gill_attachment",
"gill_spacing",
"gill_size",
"gill_color",
"stalk_shape",
"stalk_root",
"stalk_surface_above_ring",
"stalk_surface_below_ring",
"stalk_color_above_ring",
"stalk_color_below_ring",
"veil_type",
"veil_color",
"ring_number",
"ring_type",
"spore_print_color",
"population",
"habitat"
]
print(df.columns)
print(df.describe())
print(df.info())
print("Duplicates...")
print(df.duplicated().sum())

df["class"] = df["class"].map({"e":0,"p":1})
x = df.drop("class",axis=1)
y = df["class"]

#One hot Encoding...
x = pd.get_dummies(x)
print(x)
# It split into 177 columns
""""      cap_shape_b  cap_shape_c  cap_shape_f  cap_shape_k  cap_shape_s  cap_shape_x  ...  habitat_g  habitat_l  habitat_m  habitat_p  habitat_u  habitat_w
0           False        False        False        False        False         True  ...      False      False      False      False       True      False
1           False        False        False        False        False         True  ...       True      False      False      False      False      False
2            True        False        False        False        False        False  ...      False      False       True      False      False      False
3           False        False        False        False        False         True  ...      False      False      False      False       True      False
4           False        False        False        False        False         True  ...       True      False      False      False      False      False
...           ...          ...          ...          ...          ...          ...  ...        ...        ...        ...        ...        ...        ...
8119        False        False        False         True        False        False  ...      False       True      False      False      False      False
8120        False        False        False        False        False         True  ...      False       True      False      False      False      False
8121        False        False         True        False        False        False  ...      False       True      False      False      False      False
8122        False        False        False         True        False        False  ...      False       True      False      False      False      False
8123        False        False        False        False        False         True  ...      False       True      False      False      False      False

[8124 rows x 117 columns]"""

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=3123)
from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB()
model.fit(xtrain,ytrain)
predict = model.predict(xtest)
print(predict[:5])

print("Accuracy: \n",accuracy_score(ytest,predict))
print("Confusion Matrix: \n",confusion_matrix(ytest,predict))
print("classification Report :\n",classification_report(ytest,predict))
