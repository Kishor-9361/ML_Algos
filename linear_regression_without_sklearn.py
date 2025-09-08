import numpy as np
import pandas as pd

# Boston Housing dataset from UCI
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)

# Features (X) and Target (y)
X = data.drop(columns=["medv"]).values   # remove target column
y = data["medv"].values

# --- Normalize features (Standardization: mean=0, std=1) ---
X = (X - X.mean(axis=0)) / X.std(axis=0)

# --- Normalize target (optional, helps gradient descent) ---
y = (y - y.mean()) / y.std()

class LinearRegression:
    def __init__(self, learning_rate=0.001, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.m = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.m = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.m) + self.b
            dm = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.m -= self.learning_rate * dm
            self.b -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.m) + self.b

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 42)

model = LinearRegression(learning_rate=0.0001, epochs=10000)  # tuned params
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Weights (m):", model.m)
print("Bias (b):", model.b)
print("Sample Predictions:", y_pred[:10])

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
r2

import matplotlib.pyplot as plt
plt.plot(y_test, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.show()

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.show()

