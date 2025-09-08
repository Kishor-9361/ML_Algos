import numpy as np, pandas as pd

from sklearn.datasets import load_iris

iris = load_iris()
iris

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(iris.data,iris.target)

X = pd.DataFrame(iris.data,columns=iris['feature_names'])
y = pd.DataFrame(iris.target,columns=['target'])
print(X.head())
print(y.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'gini', random_state = 42)

model.fit(X_train,y_train)

from sklearn import tree
tree.plot_tree(model, filled = True)

y_pred = model.predict(X_test)
print(y_pred)
print(y_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

test_samples = [
    [5.1, 3.5, 1.4, 0.2],  # Likely Setosa -0
    [6.7, 3.1, 4.7, 1.5],  # Likely Versicolor - 1
    [7.2, 3.0, 5.8, 1.6],  # Likely Virginica - 2
    [5.1, 3.1, 1.7, 1.5]   # Likely Setosa - 0
]
Acutal = [0,1,2,0]
y_test1 = model.predict(test_samples)

print(accuracy_score(Acutal,y_test1))
print(confusion_matrix(Acutal,y_test1))
print(classification_report(Acutal,y_test1))

