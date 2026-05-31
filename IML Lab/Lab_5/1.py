import numpy as np
import pandas as pd
df = pd.read_csv("DailyDelhiClimateTest.csv")
print("missing values in dataset: ", df.isnull().sum())

df.columns = ["date","meantemp","humidity","wind_speed","meanpressure"]
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna()

X = df[["humidity", "wind_speed", "meanpressure"]].values
y = df["meantemp"].values

indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]  
y = y[indices]

mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / std

ones = np.ones((X.shape[0], 1))
X = np.hstack((ones, X))

split_index = int(0.8 * len(X))
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

XT = X_train.T
beta = np.linalg.inv(XT @ X_train) @ XT @ y_train
y_pred = X_test @ beta

ss_total = np.sum((y_test - np.mean(y_test))**2)

ss_residual = np.sum((y_test - y_pred)**2)

r2 = 1 - (ss_residual / ss_total)

mae = np.mean(np.abs(y_test - y_pred))

mse = np.mean((y_test - y_pred)**2)

rmse = np.sqrt(mse)

print("Intercept:", beta[0])
print("Coefficients:", beta[1:])
print("R2 Score:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)