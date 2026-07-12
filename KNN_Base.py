from sklearn.neighbors import KNeighborsClassifier
import numpy as np

#how the library handles the TIE , it returns the index zero from the list which is internally sotred and the List is stored alphabetically

X = np.array([[1], [2]])
y = np.array(["Male", "Female"])

model = KNeighborsClassifier(n_neighbors=2)
model.fit(X, y)

print(model.predict([[1.5]]))

# -> Output-"Female"


