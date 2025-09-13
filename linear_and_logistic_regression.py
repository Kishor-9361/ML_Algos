# Linear, Ridge, Lasso and Logistic Regression practice

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, accuracy_score

# ------------------ Boston Housing Dataset ------------------

# loading boston housing data
boston = fetch_openml(name='boston', version=1, as_frame=True)

# making dataframe
df = boston.data
df['Price'] = boston.target
print(df.head())   # just checking first few rows

# X = features, y = target
X = df.drop(columns=['Price'])
y = df['Price']

print(X.info())  # I saw some categorical columns here (CHAS, RAD)

# converting them to float (not sure if needed but I tried)
X['CHAS'] = X['CHAS'].astype(float)
X['RAD'] = X['RAD'].astype(float)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# ---------- Linear Regression ----------
lin_reg = LinearRegression()

# using cross_val_score to check error (mse but sklearn gives -mse)
mse = cross_val_score(lin_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
print("Linear Regression mse values:", mse)
print("Mean mse:", np.mean(mse))

lin_reg.fit(X_train, y_train)

# ---------- Ridge Regression ----------
ridge = Ridge()
params = {'alpha':[0.01,0.1,1,5,10,20,50]}  # just trying some values
ridge_cv = GridSearchCV(ridge, params, scoring='neg_mean_squared_error', cv=5)
ridge_cv.fit(X_train, y_train)

print("Best param ridge:", ridge_cv.best_params_)
print("Best score ridge:", ridge_cv.best_score_)

# ---------- Lasso Regression ----------
lasso = Lasso()
params = {'alpha':[0.01,0.1,1,5,10,20,50]}
lasso_cv = GridSearchCV(lasso, params, scoring='neg_mean_squared_error', cv=5)
lasso_cv.fit(X_train, y_train)

print("Best param lasso:", lasso_cv.best_params_)
print("Best score lasso:", lasso_cv.best_score_)

# checking r2 score for lasso vs linear
y_pred_lasso = lasso_cv.predict(X_test)
y_pred_lin = lin_reg.predict(X_test)

print("R2 lasso:", r2_score(y_test, y_pred_lasso))
print("R2 linear:", r2_score(y_test, y_pred_lin))

# ------------------ Logistic Regression (Breast Cancer Dataset) ------------------

cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.DataFrame(cancer.target, columns=["target"])

print(y['target'].value_counts())  # checking balance of classes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

log_reg = LogisticRegression(max_iter=200)
params = {'C':[0.1,1,10], 'max_iter':[100,200]}  # just testing some values
log_cv = GridSearchCV(log_reg, params, scoring='f1', cv=5)
log_cv.fit(X_train, y_train)

print("Best params logistic:", log_cv.best_params_)
print("Best f1 score logistic:", log_cv.best_score_)

y_pred = log_cv.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))
