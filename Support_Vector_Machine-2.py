import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data//iris.csv")
print(df.columns)
print(df.dtypes)
print(df.head())
print(df.describe())
print("Total NUll vlues.....................\n",df.isnull().sum())
print("Duplicates.......................\n",df.duplicated().sum())
# We have 3 duplicates , so that we need to drop it
df = df.drop_duplicates()
print("Duplicates After Removed...........\n",df.duplicated().sum())

"""print(df["species"].unique())
df["species"] = df["species"].map({"setosa":0,"versicolor":1,"virginica":2})
print(df)"""

x = df.drop("species",axis=1)
y = df["species"]

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=3123)

scaler = StandardScaler()  # Z = (x - ų)/std deviation --- formula for standard scaler
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)
# Now we can use the grid search method to find the best
grid = {"kernel":["linear","rbf","poly"],
        "C":[0.01,0.1,1,10,100], # if have samller value, because the margin become larger , allows some miscalssiffcation
        "gamma":["scale","auto",0.01,0.1,1,10,100]} # if larger values ,because the near by vectors only have the influence 
search = GridSearchCV(estimator=SVC(),cv=5,param_grid=grid,scoring="accuracy",n_jobs=-1,refit=True,verbose=1,return_train_score=True)

# estimator -> Machine Learning model to tune

# param_grid -> Hyperparameters and their values to test

# cv -> Number of cross-validation folds

# scoring -> Metric used to select the best model

# n_jobs -> Number of CPU cores used (-1 = use all available cores)

# refit -> Retrains the best model on the entire training dataset after GridSearchCV finishes

# verbose -> Displays the training progress during GridSearchCV

# return_train_score -> Stores training scores for analysis (default=False)

search.fit(xtrain,ytrain)
print("Best params: ",search.best_params_)
print("Best Model: ",search.best_estimator_)
print("Best Accuracy: ",search.best_score_)

model = search.best_estimator_

predict = model.predict(xtest)

print("..............TESTING...............")
print(predict[:5])
print(ytest[:5])

"""['versicolor' 'versicolor' 'setosa' 'virginica' 'setosa'] -> This is predicted
61     versicolor   -> This is acutall
86     versicolor
36         setosa
135     virginica
49         setosa
Name: species, dtype: object"""

print(f"Accuracy:{accuracy_score(ytest,predict)}")
print(f"Confusion Matrix: \n{confusion_matrix(ytest,predict)}")
print(f"Classification Report: \n{classification_report(ytest,predict)}")



"""
.....................FINAL_OUTPUT.......................

PS D:\Machine Learning> python Support_Vector_Machine.py
Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width',
       'species'],
      dtype='object')
sepal_length    float64
sepal_width     float64
petal_length    float64
petal_width     float64
species          object
dtype: object
   sepal_length  sepal_width  petal_length  petal_width species
0           5.1          3.5           1.4          0.2  setosa
1           4.9          3.0           1.4          0.2  setosa
2           4.7          3.2           1.3          0.2  setosa
3           4.6          3.1           1.5          0.2  setosa
4           5.0          3.6           1.4          0.2  setosa
       sepal_length  sepal_width  petal_length  petal_width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.054000      3.758667     1.198667
std        0.828066     0.433594      1.764420     0.763161
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000
Total NUll vlues.....................
 sepal_length    0
sepal_width     0
petal_length    0
petal_width     0
species         0
dtype: int64
Duplicates.......................
 3
Duplicates After Removed...........
 0
Fitting 5 folds for each of 105 candidates, totalling 525 fits
Best params:  {'C': 100, 'gamma': 'scale', 'kernel': 'linear'}
Best Model:  SVC(C=100, kernel='linear')
Best Accuracy:  0.9829710144927537
..............TESTING...............
['versicolor' 'versicolor' 'setosa' 'virginica' 'setosa']
61     versicolor
86     versicolor
36         setosa
135     virginica
49         setosa
Name: species, dtype: object
Accuracy:0.9666666666666667
Confusion Matrix: 
[[ 8  0  0]
 [ 0 12  0]
 [ 0  1  9]]
Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00         8
  versicolor       0.92      1.00      0.96        12
   virginica       1.00      0.90      0.95        10

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.97        30
weighted avg       0.97      0.97      0.97        30

"""