#Linear Regression , Ridge and Lasso Regression

##Data collection and imputation


import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import fetch_openml

# Load the Boston Housing dataset from OpenML
df = fetch_openml(name='boston', version=1, as_frame=True)
#as_frame= True gives a pandas dataframe

print(type(df.data))

df

#to convert the numpy array to pandas dataframe
df1 = df
df=pd.DataFrame(df.data)
df

# merging the target into the df
df['Price']= df1.target
df.head()

##divide the datasets into dependen and independent features

X = df.iloc[:, :-1]   # all columns except last(independent variables)
y = df.iloc[:, -1]    #last target column(dependent variable)

print(X.info())
#categorical data found

print(np.unique(X.RAD))
print(np.unique(X.CHAS))

X['CHAS']= X['CHAS'].astype(float)
X['RAD']= X['RAD'].astype(float)
print(X.info())

y.info()

"""splitting test and train data"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)

"""##Linear Regression"""

#Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
#makes the cross validation to get best train and test data

lin_reg = LinearRegression()
#cross validation (divides the train and test data)

#1. mse = cross_val_score(lin_reg,X,y,scoring='neg_mean_squared_error',cv=5)
mse = cross_val_score(lin_reg,X_train,y_train,scoring='neg_mean_squared_error',cv=5)
print(mse)
mse_mean = np.mean(mse)
print(mse_mean)

#note when ever the mean moves towards zero implies that performance increases
# mse of now < 1. mse

lin_reg.fit(X_train,y_train)

"""##Ridge Regression"""

#Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

#gridsearchcv is for hyperparameter

ridge = Ridge()
#parameters = {'alpha': [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20]}
parameters1 = {'alpha': [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor = GridSearchCV(ridge,parameters1,scoring= 'neg_mean_squared_error',cv=5)
#ridge_regressor.fit(X,y)
ridge_regressor.fit(X_train,y_train)

print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

##as linear regression mse is lesser than ridge(when param = parameters set)
# now also the same inference appears

#after giving values of train values the score tends to 0 slightly

"""##Lasso Regression"""

#Lasso Regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

#gridsearchcv is for hyperparameter

lasso = Lasso()
#parameters = {'alpha': [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20]}
parameters1 = {'alpha': [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor = GridSearchCV(lasso,parameters1,scoring= 'neg_mean_squared_error',cv=5)
#lasso_regressor.fit(X,y)
lasso_regressor.fit(X_train,y_train)

print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)
#linear regression mse value is lesser than lasso(when param = parameters set)

#same score and parameter appears as before

#the score and param changes with new train x and y

"""r square perfomance metric

"""

y_pred = lasso_regressor.predict(X_test)
from sklearn.metrics import r2_score

score_r2 = r2_score(y_test,y_pred)
print(score_r2)

y_pred1 = lin_reg.predict(X_test)

score_r2_1 = r2_score(y_test,y_pred1)
print(score_r2_1)

"""##Logistic Regression"""

from sklearn.linear_model import LogisticRegression

from sklearn.datasets import load_breast_cancer

df1 = load_breast_cancer()
X=pd.DataFrame(df1.data, columns=df1.feature_names)
X  #Independent Variable

y= pd.DataFrame(df1['target'],columns=["target"])
y

y.target.value_counts()

#here is the balanced data

"""Splits Testing and training data"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)

model1 = LogisticRegression(C=100,max_iter=100)

parameters = [{'C': [1,5,10]},{'max_iter': [100,150]}]
model = GridSearchCV(model1,param_grid = parameters,scoring='f1',cv=5)

model.fit(X_train,y_train)

print(model.best_params_)
print(model.best_score_)

y_predict = model.predict(X_test)
y_predict

"""##Perfomance metric for Logistic Regression"""

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
confusion_matrix(y_test,y_predict)

accuracy_score(y_test,y_predict)

print(classification_report(y_test,y_predict))