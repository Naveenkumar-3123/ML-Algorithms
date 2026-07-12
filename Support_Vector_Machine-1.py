import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data//ionosphere//ionosphere.data",sep=",",header=None)
print(df.columns) # Now here yhe columns is taken as Index values ,header=None
print(df.dtypes)
print(df.head())
print(df.describe())
print("Total NUll vlues.....................\n",df.isnull().sum())
print("Duplicates.......................\n",df.duplicated().sum())
# We have 1 duplicates , so that we need to drop it
df = df.drop_duplicates()
print("Duplicates After Removed...........\n",df.duplicated().sum())



x = df.drop(34,axis=1)
y = df[34]

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

"""
['b' 'b' 'g' 'g' 'g']
230    b
196    b
56     g
281    g
339    g
Name: 34, dtype: object
"""

print(f"Accuracy:{accuracy_score(ytest,predict)}")
print(f"Confusion Matrix: \n{confusion_matrix(ytest,predict)}")
print(f"Classification Report: \n{classification_report(ytest,predict)}")



"""...........................FINAL OUTPUT.................................
Index([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
      dtype='int64')
0       int64
1       int64
2     float64
3     float64
4     float64
5     float64
6     float64
7     float64
8     float64
9     float64
10    float64
11    float64
12    float64
13    float64
14    float64
15    float64
16    float64
17    float64
18    float64
19    float64
20    float64
21    float64
22    float64
23    float64
24    float64
25    float64
26    float64
27    float64
28    float64
29    float64
30    float64
31    float64
32    float64
33    float64
34     object
dtype: object
   0   1        2        3        4        5        6        7        8   ...       26       27       28       29       30       31       32       33  34
0   1   0  0.99539 -0.05889  0.85243  0.02306  0.83398 -0.37708  1.00000  ...  0.41078 -0.46168  0.21266 -0.34090  0.42267 -0.54487  0.18641 -0.45300   g
1   1   0  1.00000 -0.18829  0.93035 -0.36156 -0.10868 -0.93597  1.00000  ... -0.20468 -0.18401 -0.19040 -0.11593 -0.16626 -0.06288 -0.13738 -0.02447   b
2   1   0  1.00000 -0.03365  1.00000  0.00485  1.00000 -0.12062  0.88965  ...  0.58984 -0.22145  0.43100 -0.17365  0.60436 -0.24180  0.56045 -0.38238   g
3   1   0  1.00000 -0.45161  1.00000  1.00000  0.71216 -1.00000  0.00000  ...  0.51613  1.00000  1.00000 -0.20099  0.25682  1.00000 -0.32382  1.00000   b
4   1   0  1.00000 -0.02401  0.94140  0.06531  0.92106 -0.23255  0.77152  ...  0.13290 -0.53206  0.02431 -0.62197 -0.05707 -0.59573 -0.04608 -0.65697   g

[5 rows x 35 columns]
               0      1           2           3           4           5   ...          28          29          30          31          32          33
count  351.000000  351.0  351.000000  351.000000  351.000000  351.000000  ...  351.000000  351.000000  351.000000  351.000000  351.000000  351.000000
mean     0.891738    0.0    0.641342    0.044372    0.601068    0.115889  ...    0.378445   -0.027907    0.352514   -0.003794    0.349364    0.014480
std      0.311155    0.0    0.497708    0.441435    0.519862    0.460810  ...    0.575886    0.507974    0.571483    0.513574    0.522663    0.468337
min      0.000000    0.0   -1.000000   -1.000000   -1.000000   -1.000000  ...   -1.000000   -1.000000   -1.000000   -1.000000   -1.000000   -1.000000
25%      1.000000    0.0    0.472135   -0.064735    0.412660   -0.024795  ...    0.000000   -0.236885    0.000000   -0.242595    0.000000   -0.165350
50%      1.000000    0.0    0.871110    0.016310    0.809200    0.022800  ...    0.496640    0.000000    0.442770    0.000000    0.409560    0.000000
75%      1.000000    0.0    1.000000    0.194185    1.000000    0.334655  ...    0.883465    0.154075    0.857620    0.200120    0.813765    0.171660
max      1.000000    0.0    1.000000    1.000000    1.000000    1.000000  ...    1.000000    1.000000    1.000000    1.000000    1.000000    1.000000

[8 rows x 34 columns]
Total NUll vlues.....................
 0     0
1     0
2     0
3     0
4     0
5     0
6     0
7     0
8     0
9     0
10    0
11    0
12    0
13    0
14    0
15    0
16    0
17    0
18    0
19    0
20    0
21    0
22    0
23    0
24    0
25    0
26    0
27    0
28    0
29    0
30    0
31    0
32    0
33    0
34    0
dtype: int64
Duplicates.......................
 1
Duplicates After Removed...........
 0
Fitting 5 folds for each of 105 candidates, totalling 525 fits
Best params:  {'C': 1, 'gamma': 0.1, 'kernel': 'rbf'}
Best Model:  SVC(C=1, gamma=0.1)
Best Accuracy:  0.95
..............TESTING...............
['b' 'b' 'g' 'g' 'g']
230    b
196    b
56     g
281    g
339    g
Name: 34, dtype: object
Accuracy:0.9571428571428572
Confusion Matrix: 
[[24  1]
 [ 2 43]]
Classification Report: 
              precision    recall  f1-score   support

           b       0.92      0.96      0.94        25
           g       0.98      0.96      0.97        45

    accuracy                           0.96        70
   macro avg       0.95      0.96      0.95        70
weighted avg       0.96      0.96      0.96        70

"""